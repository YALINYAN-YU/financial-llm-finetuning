"""
V3 — Retrieve relevant chunks from the FAISS knowledge base index.

Run from project root:
    python src/rag_retrieve.py "Apple reported record earnings"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_DIR = Path("rag_index")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_project_root() -> None:
    if not Path("src/rag_retrieve.py").exists():
        print("Error: Run this script from the project root.")
        sys.exit(1)


def check_index(index_dir: Path) -> None:
    required = [index_dir / "index.faiss", index_dir / "chunks.json", index_dir / "meta.json"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Error: RAG index not found.\n")
        for p in missing:
            print(f"  Missing: {p}")
        print("\nRun this command first:")
        print("  python src/build_rag_index.py")
        sys.exit(1)


class RAGRetriever:
    """Load FAISS index and retrieve top-k relevant knowledge-base chunks."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        check_index(index_dir)
        self.index_dir = index_dir

        with open(index_dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)

        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.model = SentenceTransformer(self.meta["embedding_model"])

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k chunks ranked by cosine similarity."""
        embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embedding)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )
        return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks as a single context block."""
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[{i}] ({r['source']}, score={r['score']:.3f})\n{r['text']}")
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve from financial RAG index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    ensure_project_root()
    args = parse_args()

    retriever = RAGRetriever(args.index_dir)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    print("=" * 55)
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i} — {r['source']} (score={r['score']:.3f})")
        print("-" * 55)
        print(r["text"])


if __name__ == "__main__":
    main()
