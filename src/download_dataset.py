"""Download Financial PhraseBank from Hugging Face and save train/val/test splits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


# Parquet-based dataset — no legacy loading script, works on Colab/modern datasets
DATASET_NAME = "lmassaron/FinancialPhraseBank"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
INSTRUCTION = (
    "Classify the sentiment of the following financial news sentence "
    "as negative, neutral, or positive."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Financial PhraseBank and create train/validation/test splits."
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


def load_financial_phrasebank():
    """Load Financial PhraseBank from Hugging Face (Parquet, no loading script)."""
    dataset_dict = load_dataset(DATASET_NAME)
    parts = [dataset_dict[split] for split in ("train", "validation", "test") if split in dataset_dict]
    if not parts:
        raise ValueError(f"No splits found in dataset {DATASET_NAME}")

    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def format_example(example: dict) -> dict:
    label_id = int(example["label"])
    sentiment = example.get("sentiment") or LABEL_NAMES[label_id]
    return {
        "sentence": example["sentence"],
        "label": label_id,
        "instruction": INSTRUCTION,
        "output": sentiment,
    }


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
    print(f"Source         : {DATASET_NAME}")
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
        formatted = split.map(format_example, remove_columns=split.column_names)
        path = output_dir / f"{name}.jsonl"
        formatted.to_json(path, orient="records", lines=True)
        print(f"Saved {name:>10} -> {path}")


def main() -> None:
    args = parse_args()

    print(f"Downloading {DATASET_NAME}...")
    dataset = load_financial_phrasebank()

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
