"""
QLoRA fine-tuning script for financial sentiment classification.

Fine-tunes Qwen2.5-Instruct on instruction-formatted Financial PhraseBank data
using 4-bit quantization (BitsAndBytes) and LoRA adapters (PEFT), orchestrated
via TRL's SFTTrainer.

Designed for GPU environments (Colab, Kaggle, cloud VMs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN_FILE = Path("data/train.jsonl")
DEFAULT_VAL_FILE = Path("data/validation.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/model")

# LoRA hyperparameters — rank/alpha control adapter capacity vs. memory
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

# Target every linear projection in Qwen2.5 attention + MLP blocks
QWEN_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with production-ready defaults."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5 on financial instruction data"
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Effective batch = batch_size x grad_accum x num_gpus",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    """
    Prefer bfloat16 on hardware that supports it (A100, H100, RTX 30xx+).
    Fall back to float16 on older CUDA GPUs; use float32 on CPU (not recommended).
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_datasets(train_file: Path, val_file: Path):
    """
    Load train and validation JSONL files produced by download_dataset.py.

    Each record contains:
        sentence, label, instruction, output
    """
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    data_files = {"train": str(train_file), "validation": str(val_file)}
    dataset = load_dataset("json", data_files=data_files)

    print(f"Train examples     : {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    return dataset["train"], dataset["validation"]


def to_conversational_format(example: dict) -> dict:
    """
    Convert a flat JSONL record into TRL's conversational schema.

    TRL requires a `messages` column for assistant_only_loss=True.
    Using formatting_func together with assistant_only_loss is unsupported
    in recent TRL versions, so we preprocess the dataset instead.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{example['instruction']}\n\n"
                    f"Sentence: {example['sentence']}"
                ),
            },
            {
                "role": "assistant",
                "content": example["output"],
            },
        ]
    }


def prepare_conversational_dataset(dataset):
    """Map raw JSONL fields to conversational `messages` and drop unused columns."""
    columns_to_remove = [col for col in dataset.column_names if col != "messages"]
    return dataset.map(
        to_conversational_format,
        remove_columns=columns_to_remove,
        desc="Formatting as conversational messages",
    )


def load_tokenizer(model_name: str):
    """Load the Qwen2.5 tokenizer and ensure padding is configured for batching."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen models often lack an explicit pad token; reuse EOS for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def load_quantized_model(model_name: str, compute_dtype: torch.dtype):
    """
    Load the base model in 4-bit NF4 precision (QLoRA).

    - load_in_4bit:          store frozen weights in 4-bit, cutting VRAM ~4x
    - bnb_4bit_quant_type:   NormalFloat4 — optimal for normally-distributed weights
    - bnb_4bit_use_double_quant: quantize the quantization constants for extra savings
    - device_map="auto":     spread layers across available GPUs automatically
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # Enable gradient checkpointing hooks required for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    return model


def build_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    """
    Configure LoRA adapters injected into attention and MLP projections.

    Only ~0.1–1% of parameters are trainable; the 4-bit base weights stay frozen.
    """
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_LORA_TARGET_MODULES,
    )


def build_training_args(
    output_dir: Path,
    compute_dtype: torch.dtype,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> SFTConfig:
    """
    Build TRL SFTConfig (extends HuggingFace TrainingArguments).

    Effective batch size = batch_size x gradient_accumulation_steps x num_gpus
                         = 1 x 4 x 1 = 4 (smoke-test defaults, single GPU)

    max_steps=20 caps training early so we can verify the full pipeline
    (dataset → tokenizer → model → LoRA → trainer → save) before a full run.
    """
    use_bf16 = compute_dtype == torch.bfloat16
    use_fp16 = compute_dtype == torch.float16 and torch.cuda.is_available()

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        # Use bf16 when hardware supports it; otherwise fp16 on CUDA
        bf16=use_bf16,
        fp16=use_fp16,
        # Log and evaluate every epoch
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Reproducibility and memory
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Sequence length for SFTTrainer tokenization
        max_length=max_length,
        # Mask loss to assistant tokens only (don't penalize user/instruction tokens)
        assistant_only_loss=True,
        # Disable external loggers in headless environments
        report_to="none",
        # Dataloader workers — 0 is safest across Colab/Kaggle/local
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def print_trainable_parameters(model) -> None:
    """Log the fraction of parameters that are actually updated during training."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = get_compute_dtype()
    print(f"Model          : {args.model_name}")
    print(f"Compute dtype  : {compute_dtype}")
    print(f"Output dir     : {args.output_dir}")

    # ------------------------------------------------------------------
    # 1. Dataset loading + conversational formatting
    # ------------------------------------------------------------------
    train_dataset, eval_dataset = load_datasets(args.train_file, args.val_file)
    train_dataset = prepare_conversational_dataset(train_dataset)
    eval_dataset = prepare_conversational_dataset(eval_dataset)

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(args.model_name)

    # ------------------------------------------------------------------
    # 3. Quantized base model (4-bit) + LoRA adapters (QLoRA)
    # ------------------------------------------------------------------
    model = load_quantized_model(args.model_name, compute_dtype)
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # 4. Training configuration
    # ------------------------------------------------------------------
    training_args = build_training_args(
        output_dir=args.output_dir,
        compute_dtype=compute_dtype,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. Training loop (TRL SFTTrainer handles tokenization + collation)
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Model saving — adapter weights + tokenizer
    # ------------------------------------------------------------------
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Training complete.")


if __name__ == "__main__":
    main()
