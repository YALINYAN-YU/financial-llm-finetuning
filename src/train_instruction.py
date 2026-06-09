"""
V2 — QLoRA instruction tuning for financial sentiment analysis.

Fine-tunes Qwen2.5-Instruct on rich instruction-response pairs where the model
learns to classify sentiment AND provide natural-language reasoning.

Pipeline position:
    create_instruction_dataset.py  →  train_instruction.py  →  results/model-instruction

Input:  data/instruction/train.jsonl, data/instruction/validation.jsonl
Output: results/model-instruction/
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
# Defaults — tuned for V2 instruction data (longer responses than V1)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN_FILE = Path("data/instruction/train.jsonl")
DEFAULT_VAL_FILE = Path("data/instruction/validation.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/model-instruction")

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

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
    parser = argparse.ArgumentParser(
        description="V2 QLoRA instruction tuning on financial sentiment data"
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # Longer max_length to accommodate Sentiment + Reason responses
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=20, help="Cap steps for smoke tests")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_datasets(train_file: Path, val_file: Path):
    """
    Load V2 instruction JSONL files.

    Each record:
        instruction, input, output, sentence, label, sentiment
    """
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Run: python src/create_instruction_dataset.py"
        )
    if not val_file.exists():
        raise FileNotFoundError(
            f"Validation file not found: {val_file}\n"
            "Run: python src/create_instruction_dataset.py"
        )

    data_files = {"train": str(train_file), "validation": str(val_file)}
    dataset = load_dataset("json", data_files=data_files)

    print(f"Train examples     : {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    return dataset["train"], dataset["validation"]


def to_conversational_format(example: dict) -> dict:
    """
    Map instruction + input → user turn, rich output → assistant turn.

    Example user message:
        Analyze the sentiment... \n\n Apple reported record earnings.

    Example assistant message:
        Sentiment: Positive\n\nReason:\nThe company exceeded...
    """
    user_content = example["instruction"]
    if example.get("input"):
        user_content = f"{user_content}\n\n{example['input']}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def prepare_conversational_dataset(dataset):
    columns_to_remove = [col for col in dataset.column_names if col != "messages"]
    return dataset.map(
        to_conversational_format,
        remove_columns=columns_to_remove,
        desc="Formatting instruction data as conversations",
    )


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_model(model_name: str, compute_dtype: torch.dtype):
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

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return model


def build_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
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
    max_steps: int,
    seed: int,
) -> SFTConfig:
    use_bf16 = compute_dtype == torch.bfloat16
    use_fp16 = compute_dtype == torch.float16 and torch.cuda.is_available()

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        assistant_only_loss=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def print_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = get_compute_dtype()
    print("=" * 55)
    print("V2 Financial Instruction Tuning")
    print("=" * 55)
    print(f"Model          : {args.model_name}")
    print(f"Compute dtype  : {compute_dtype}")
    print(f"Train file     : {args.train_file}")
    print(f"Val file       : {args.val_file}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Max steps      : {args.max_steps}")

    # 1. Load V2 instruction dataset
    train_dataset, eval_dataset = load_datasets(args.train_file, args.val_file)
    train_dataset = prepare_conversational_dataset(train_dataset)
    eval_dataset = prepare_conversational_dataset(eval_dataset)

    # 2. Tokenizer
    tokenizer = load_tokenizer(args.model_name)

    # 3. 4-bit model + LoRA (QLoRA)
    model = load_quantized_model(args.model_name, compute_dtype)
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # 4. Training config
    training_args = build_training_args(
        output_dir=args.output_dir,
        compute_dtype=compute_dtype,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # 5. Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting V2 QLoRA instruction tuning...")
    trainer.train()

    # 6. Save adapter + tokenizer
    print(f"\nSaving instruction-tuned adapter to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("V2 training complete.")


if __name__ == "__main__":
    main()
