"""
V2 — Run inference with the instruction-tuned LoRA adapter.

Loads the base Qwen2.5 model + LoRA weights from results/model-instruction
and generates rich sentiment responses (Sentiment + Reason) for sample inputs.

Pipeline position:
    train_instruction.py  →  infer_instruction.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_PATH = Path("results/model-instruction")

INSTRUCTION = (
    "Analyze the sentiment of the following financial news sentence. "
    "Classify it as negative, neutral, or positive, and explain your reasoning."
)

# Three financial sentiment examples: positive, negative, neutral
DEMO_EXAMPLES = [
    {
        "input": "Apple reported record earnings.",
        "expected_sentiment": "positive",
    },
    {
        "input": "Sales by Seppala diminished by 6 per cent.",
        "expected_sentiment": "negative",
    },
    {
        "input": "The purchase sum is about EUR 10mn (US$ 12.97 mn).",
        "expected_sentiment": "neutral",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2 inference with instruction-tuned LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to LoRA adapter (results/model-instruction)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy decoding (recommended for evaluation)",
    )
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(base_model: str, adapter_path: Path):
    """
    Load 4-bit quantized base model and merge LoRA adapter weights.

    The adapter directory contains adapter_config.json pointing to the base
    model; the tokenizer is loaded from the adapter path (saved during training).
    """
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}\n"
            "Run: python src/train_instruction.py"
        )

    compute_dtype = get_compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def build_user_message(instruction: str, user_input: str) -> str:
    return f"{instruction}\n\n{user_input}"


def generate_response(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate assistant response using the Qwen2.5 chat template."""
    messages = [{"role": "user", "content": build_user_message(instruction, user_input)}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def run_demo(model, tokenizer, max_new_tokens: int, temperature: float) -> None:
    print("=" * 60)
    print("V2 Instruction Inference — Financial Sentiment Examples")
    print("=" * 60)

    for i, example in enumerate(DEMO_EXAMPLES, start=1):
        user_input = example["input"]
        expected = example["expected_sentiment"]

        print(f"\nExample {i}")
        print("-" * 60)
        print(f"Input            : {user_input}")
        print(f"Expected sentiment: {expected}")
        print("-" * 60)

        response = generate_response(
            model,
            tokenizer,
            INSTRUCTION,
            user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        print("Model response:")
        print(response)
        print("-" * 60)


def main() -> None:
    args = parse_args()

    print(f"Base model    : {args.base_model}")
    print(f"Adapter path  : {args.adapter_path}")
    print(f"Compute dtype : {get_compute_dtype()}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)
    run_demo(model, tokenizer, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
