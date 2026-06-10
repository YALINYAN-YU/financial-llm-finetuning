"""
V3 — Build a FAISS retrieval index from the financial knowledge base.

Loads markdown files from data/knowledge_base/, chunks the text,
embeds chunks with sentence-transformers, and saves the index to rag_index/.

Run from project root:
    python src/build_rag_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_KB_DIR = Path("data/knowledge_base")
DEFAULT_INDEX_DIR = Path("rag_index")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def ensure_project_root() -> None:
    if not Path("src/build_rag_index.py").exists():
        print("Error: Run this script from the project root.")
        print("  cd financial-llm-finetuning")
        print("  python src/build_rag_index.py")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from knowledge base")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    return parser.parse_args()


def load_markdown_files(kb_dir: Path) -> list[dict]:
    """Load all markdown files from the knowledge base directory."""
    if not kb_dir.exists():
        print(f"Error: Knowledge base not found: {kb_dir}")
        sys.exit(1)

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        print(f"Error: No .md files found in {kb_dir}")
        sys.exit(1)

    documents = []
    for path in md_files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": path.name, "text": text})
            print(f"  Loaded {path.name} ({len(text):,} chars)")
    return documents


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        else:
            if buffer:
                chunks.append({"source": source, "text": buffer})
            if len(para) <= chunk_size:
                buffer = para
            else:
                # Fall back to fixed-size windows for long paragraphs
                start = 0
                while start < len(para):
                    piece = para[start : start + chunk_size]
                    chunks.append({"source": source, "text": piece})
                    start += chunk_size - overlap
                buffer = ""

    if buffer:
        chunks.append({"source": source, "text": buffer})

    return chunks


def build_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    all_chunks: list[dict] = []
    for doc in documents:
        pieces = chunk_text(doc["text"], doc["source"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "id": len(all_chunks),
                    "source": piece["source"],
                    "text": piece["text"],
                }
            )
    return all_chunks


def embed_chunks(chunks: list[dict], model_name: str) -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def save_index(chunks: list[dict], embeddings: np.ndarray, index_dir: Path, model_name: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    meta = {
        "embedding_model": model_name,
        "num_chunks": len(chunks),
        "dimension": dim,
        "sources": sorted({c["source"] for c in chunks}),
    }
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved FAISS index  -> {index_dir / 'index.faiss'}")
    print(f"Saved chunk store  -> {index_dir / 'chunks.json'}")
    print(f"Saved metadata     -> {index_dir / 'meta.json'}")
    print(f"Total chunks       : {len(chunks)}")


def main() -> None:
    ensure_project_root()
    args = parse_args()

    print("=" * 55)
    print("V3 — Build RAG Index")
    print("=" * 55)
    print(f"Knowledge base : {args.kb_dir}")
    print(f"Output dir     : {args.index_dir}")

    print("\nLoading markdown files:")
    documents = load_markdown_files(args.kb_dir)

    chunks = build_chunks(documents, args.chunk_size, args.chunk_overlap)
    print(f"\nCreated {len(chunks)} chunks from {len(documents)} documents")

    embeddings = embed_chunks(chunks, args.model)
    save_index(chunks, embeddings, args.index_dir, args.model)

    print("\nRAG index ready. Next step:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
