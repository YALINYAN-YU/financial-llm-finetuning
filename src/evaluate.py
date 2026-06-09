"""Evaluate a fine-tuned language model on a held-out dataset."""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import build_prompt, generate


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return matches / len(predictions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLM")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to fine-tuned model")
    parser.add_argument("--eval-file", type=Path, required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to evaluate")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    dataset = load_dataset("json", data_files=str(args.eval_file), split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    predictions = []
    references = []
    records = []

    for example in tqdm(dataset, desc="Evaluating"):
        instruction = example.get("instruction", "")
        user_input = example.get("input", "")
        reference = example.get("output", "")

        prompt = build_prompt(instruction, user_input)
        prediction = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)

        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "instruction": instruction,
                "input": user_input,
                "reference": reference,
                "prediction": prediction,
            }
        )

    exact_match = compute_exact_match(predictions, references)
    metrics = {"exact_match": exact_match, "num_examples": len(predictions)}

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.jsonl"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(predictions_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Exact match: {exact_match:.4f} ({len(predictions)} examples)")
    print(f"Metrics saved to {metrics_path}")
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
