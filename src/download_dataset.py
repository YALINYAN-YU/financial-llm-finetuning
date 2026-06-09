"""Download Financial PhraseBank from Hugging Face and save train/val/test splits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "takala/financial_phrasebank"
DEFAULT_CONFIG = "sentences_allagree"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Financial PhraseBank and create train/validation/test splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        choices=[
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        ],
        help="Annotator agreement threshold configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store dataset splits",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


def label_distribution(dataset) -> dict[str, int]:
    counts = Counter(dataset["label"])
    return {LABEL_NAMES[label_id]: counts[label_id] for label_id in sorted(counts)}


def print_split_stats(name: str, dataset) -> None:
    dist = label_distribution(dataset)
    print(f"\n{name} ({len(dataset)} examples)")
    print("-" * 40)
    for label, count in dist.items():
        pct = 100.0 * count / len(dataset)
        print(f"  {label:>8}: {count:>5} ({pct:5.1f}%)")


def print_dataset_statistics(splits: dict) -> None:
    total = sum(len(split) for split in splits.values())

    print("=" * 50)
    print("Financial PhraseBank — Dataset Statistics")
    print("=" * 50)
    print(f"Total examples : {total}")
    print(f"Splits         : {', '.join(splits)}")

    for name, split in splits.items():
        print_split_stats(name.capitalize(), split)

    print("\nOverall label distribution")
    print("-" * 40)
    all_labels = []
    for split in splits.values():
        all_labels.extend(split["label"])

    overall = Counter(all_labels)
    for label_id in sorted(overall):
        count = overall[label_id]
        pct = 100.0 * count / total
        print(f"  {LABEL_NAMES[label_id]:>8}: {count:>5} ({pct:5.1f}%)")


def create_splits(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    validate_ratios(train_ratio, val_ratio, test_ratio)

    train_test = dataset.train_test_split(test_size=(1.0 - train_ratio), seed=seed)
    train_split = train_test["train"]
    holdout = train_test["test"]

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_test = holdout.train_test_split(test_size=(1.0 - relative_val), seed=seed)

    return {
        "train": train_split,
        "validation": val_test["train"],
        "test": val_test["test"],
    }


def save_splits(splits: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        path = output_dir / f"{name}.jsonl"
        split.to_json(path, orient="records", lines=True)
        print(f"Saved {name:>10} -> {path}")


def main() -> None:
    args = parse_args()

    print(f"Downloading {DATASET_NAME} (config: {args.config})...")
    dataset = load_dataset(DATASET_NAME, args.config, split="train", trust_remote_code=True)

    print(f"Loaded {len(dataset)} examples from Hugging Face.")
    splits = create_splits(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_splits(splits, args.output_dir)
    print_dataset_statistics(splits)


if __name__ == "__main__":
    main()
