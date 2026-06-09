"""Run inference with a fine-tuned language model."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(instruction: str, user_input: str = "") -> str:
    if user_input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in generated:
        return generated.split("### Response:")[-1].strip()
    return generated[len(prompt) :].strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LLM")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, required=True, help="Instruction or question")
    parser.add_argument("--input", type=str, default="", help="Optional additional context")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    prompt = build_prompt(args.prompt, args.input)
    response = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)

    print(response)


if __name__ == "__main__":
    main()
