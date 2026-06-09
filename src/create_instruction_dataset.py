"""
V2 — Convert Financial PhraseBank into rich instruction-tuning data.

Transforms short sentiment labels into structured responses with classification
and natural-language reasoning, suitable for financial instruction tuning.

Pipeline position:
    download_dataset.py  →  create_instruction_dataset.py  →  train_instruction.py

Input  (V1):  data/train.jsonl, data/validation.jsonl, data/test.jsonl
Output (V2):  data/instruction/train.jsonl, validation.jsonl, test.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
HF_DATASET_NAME = "lmassaron/FinancialPhraseBank"

INSTRUCTION = (
    "Analyze the sentiment of the following financial news sentence. "
    "Classify it as negative, neutral, or positive, and explain your reasoning."
)

# Keyword hints used to generate contextual reasons from the sentence text
POSITIVE_KEYWORDS = (
    "profit", "up", "rose", "increased", "growth", "record", "exceeded",
    "strong", "gain", "higher", "surge", "boost", "improved", "expand",
)
NEGATIVE_KEYWORDS = (
    "loss", "down", "fell", "declined", "diminished", "decreased", "missed",
    "weak", "drop", "cut", "lower", "slump", "deficit", "layoff", "warning",
)
NEUTRAL_KEYWORDS = (
    "reported", "announced", "said", "according", "stated", "filed",
    "scheduled", "expected", "plan", "agreed",
)

SPLIT_FILES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create V2 instruction-tuning dataset from Financial PhraseBank."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing V1 JSONL splits (train/validation/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/instruction"),
        help="Directory to write V2 instruction JSONL files",
    )
    parser.add_argument(
        "--from-huggingface",
        action="store_true",
        help="Download from Hugging Face when local V1 files are missing",
    )
    parser.add_argument(
        "--write-examples",
        type=Path,
        default=Path("examples/instruction_examples.jsonl"),
        help="Path to write curated example prompts for documentation",
    )
    return parser.parse_args()


def resolve_sentiment(example: dict) -> str:
    """Return sentiment string from label id or existing sentiment field."""
    if "sentiment" in example and example["sentiment"]:
        return str(example["sentiment"]).lower()
    label_id = int(example["label"])
    return LABEL_NAMES[label_id]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def generate_reason(sentence: str, sentiment: str) -> str:
    """
    Build a short, sentence-aware explanation for the assigned sentiment.

    Uses keyword heuristics over the financial sentence to produce varied,
    contextually grounded reasons instead of a single static template.
    """
    text = sentence.lower()

    if sentiment == "positive":
        if _contains_any(text, ("record", "exceeded", "beat", "surpassed")):
            return (
                "The company exceeded earnings expectations and reported "
                "strong financial performance."
            )
        if _contains_any(text, ("profit", "earnings", "revenue", "sales")) and _contains_any(
            text, ("up", "rose", "increased", "growth", "higher", "gain")
        ):
            return (
                "The text signals favorable financial outcomes, with rising "
                "profits, earnings, or sales indicating positive momentum."
            )
        if _contains_any(text, POSITIVE_KEYWORDS):
            return (
                "The language reflects positive momentum in the company's "
                "financial metrics or market outlook."
            )
        return (
            "The statement conveys an optimistic tone regarding the "
            "company's financial position or performance."
        )

    if sentiment == "negative":
        if _contains_any(text, ("loss", "deficit", "debt", "default")):
            return (
                "The text highlights adverse financial developments such as "
                "losses, debt pressure, or deteriorating performance."
            )
        if _contains_any(text, ("down", "fell", "declined", "diminished", "cut")):
            return (
                "The language points to weakening results, with declining "
                "figures or unfavorable business developments."
            )
        if _contains_any(text, NEGATIVE_KEYWORDS):
            return (
                "The statement suggests negative implications for the "
                "company's financial health or investor sentiment."
            )
        return (
            "The statement conveys a pessimistic tone that may weigh on "
            "perceptions of the company's financial outlook."
        )

    # neutral
    if _contains_any(text, NEUTRAL_KEYWORDS):
        return (
            "The text presents factual financial information without clear "
            "positive or negative market implications."
        )
    return (
        "The statement is largely descriptive and does not express a strong "
        "bullish or bearish view on the company's finances."
    )


def format_rich_response(sentiment: str, reason: str) -> str:
    """Format the assistant response with sentiment label and explanation."""
    return f"Sentiment: {sentiment.capitalize()}\n\nReason:\n{reason}"


def to_instruction_record(example: dict) -> dict:
    """Convert a V1 PhraseBank record into a V2 instruction-tuning record."""
    sentence = example["sentence"]
    sentiment = resolve_sentiment(example)
    reason = generate_reason(sentence, sentiment)

    return {
        "instruction": INSTRUCTION,
        "input": sentence,
        "output": format_rich_response(sentiment, reason),
        "sentence": sentence,
        "label": int(example["label"]),
        "sentiment": sentiment,
    }


def load_split_from_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split_from_huggingface(split_name: str) -> list[dict]:
    dataset = load_dataset(HF_DATASET_NAME, split=split_name)
    return [dict(row) for row in dataset]


def load_v1_split(input_dir: Path, split_name: str, from_hf: bool) -> list[dict]:
    path = input_dir / f"{split_name}.jsonl"
    if path.exists():
        print(f"Loading {split_name} from {path}")
        return load_split_from_jsonl(path)

    if from_hf:
        print(f"Loading {split_name} from Hugging Face ({HF_DATASET_NAME})")
        return load_split_from_huggingface(split_name)

    raise FileNotFoundError(
        f"Missing {path}. Run `python src/download_dataset.py` first, "
        "or pass --from-huggingface."
    )


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records):>5} examples -> {path}")


def build_curated_examples() -> list[dict]:
    """Hand-crafted examples written to examples/instruction_examples.jsonl."""
    examples = [
        (
            "Apple reported record earnings.",
            "positive",
            2,
            "The company exceeded earnings expectations and reported "
            "strong financial performance.",
        ),
        (
            "Sales by Seppala diminished by 6 per cent.",
            "negative",
            0,
            "The language points to weakening results, with declining "
            "figures or unfavorable business developments.",
        ),
        (
            "The purchase sum is about EUR 10mn (US$ 12.97 mn).",
            "neutral",
            1,
            "The text presents factual financial information without clear "
            "positive or negative market implications.",
        ),
        (
            "Operating profit was EUR 139.7 mn, up 23% from EUR 113.8 mn.",
            "positive",
            2,
            "The text signals favorable financial outcomes, with rising "
            "profits, earnings, or sales indicating positive momentum.",
        ),
        (
            "The company warned that full-year guidance may be revised downward.",
            "negative",
            0,
            "The statement suggests negative implications for the "
            "company's financial health or investor sentiment.",
        ),
    ]

    return [
        {
            "instruction": INSTRUCTION,
            "input": sentence,
            "output": format_rich_response(sentiment, reason),
            "sentence": sentence,
            "label": label,
            "sentiment": sentiment,
        }
        for sentence, sentiment, label, reason in examples
    ]


def print_statistics(splits: dict[str, list[dict]]) -> None:
    total = sum(len(records) for records in splits.values())
    print("\n" + "=" * 55)
    print("V2 Instruction Dataset — Statistics")
    print("=" * 55)
    print(f"Total examples : {total}")

    for name, records in splits.items():
        counts = Counter(r["sentiment"] for r in records)
        print(f"\n{name.capitalize()} ({len(records)} examples)")
        print("-" * 40)
        for sentiment in ("negative", "neutral", "positive"):
            count = counts.get(sentiment, 0)
            pct = 100.0 * count / len(records) if records else 0.0
            print(f"  {sentiment:>8}: {count:>5} ({pct:5.1f}%)")


def print_sample(records: list[dict], n: int = 2) -> None:
    print("\nSample V2 records:")
    print("-" * 55)
    for record in records[:n]:
        print(f"Input : {record['input']}")
        print(f"Output: {record['output']}")
        print("-" * 55)


def main() -> None:
    args = parse_args()

    splits: dict[str, list[dict]] = {}
    for split_name in SPLIT_FILES:
        v1_records = load_v1_split(args.input_dir, split_name, args.from_huggingface)
        splits[split_name] = [to_instruction_record(row) for row in v1_records]

    for split_name, records in splits.items():
        save_jsonl(records, args.output_dir / f"{split_name}.jsonl")

    save_jsonl(build_curated_examples(), args.write_examples)

    print_statistics(splits)
    print_sample(splits["train"])


if __name__ == "__main__":
    main()
