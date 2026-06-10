"""
V3 — RAG-powered Financial AI Assistant (Streamlit demo).

Combines knowledge-base retrieval with the V2 instruction-tuned model
to analyze financial news and display sentiment + reasoning.

Run from project root:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.infer_instruction import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_BASE_MODEL,
    INSTRUCTION,
    generate_response,
    get_compute_dtype,
    load_model_and_tokenizer,
)
from src.rag_retrieve import RAGRetriever, format_context


ADAPTER_PATH = DEFAULT_ADAPTER_PATH
BASE_MODEL = DEFAULT_BASE_MODEL
INDEX_DIR = Path("rag_index")


def adapter_exists(path: Path = ADAPTER_PATH) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def index_exists(path: Path = INDEX_DIR) -> bool:
    return all((path / f).exists() for f in ("index.faiss", "chunks.json", "meta.json"))


@st.cache_resource(show_spinner="Loading V2 instruction model...")
def load_instruction_model():
    return load_model_and_tokenizer(BASE_MODEL, ADAPTER_PATH)


@st.cache_resource(show_spinner="Loading RAG index...")
def load_retriever():
    return RAGRetriever(INDEX_DIR)


def build_augmented_input(user_text: str, context: str) -> str:
    """Combine retrieved knowledge with the user's financial sentence."""
    return (
        f"Reference context from financial knowledge base:\n{context}\n\n"
        f"Financial news sentence to analyze:\n{user_text}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Financial AI Assistant",
        page_icon="📊",
        layout="wide",
    )

    st.title("Financial AI Assistant")
    st.caption("V3 — RAG + Instruction-Tuned Sentiment Analysis (Qwen2.5 + QLoRA)")

    # --- Prerequisites ---
    col1, col2 = st.columns(2)
    with col1:
        rag_ok = index_exists()
        st.metric("RAG Index", "Ready" if rag_ok else "Missing")
    with col2:
        model_ok = adapter_exists()
        st.metric("V2 Adapter", "Ready" if model_ok else "Missing")

    if not rag_ok:
        st.warning(
            "RAG index not found. Build it first:\n\n"
            "`python src/build_rag_index.py`"
        )
        return

    if not model_ok:
        st.warning(
            "V2 LoRA adapter not found. Train the instruction model first:\n\n"
            "```\n"
            "python src/download_dataset.py\n"
            "python src/create_instruction_dataset.py\n"
            "python src/train_instruction.py\n"
            "```"
        )

    # --- Input ---
    user_input = st.text_area(
        "Enter financial news or a question",
        placeholder="e.g. Apple reported record earnings.",
        height=100,
    )
    top_k = st.slider("Retrieved context chunks", min_value=1, max_value=5, value=3)
    run_btn = st.button("Analyze", type="primary")

    if not run_btn:
        st.info("Enter a financial sentence and click **Analyze**.")
        return

    if not user_input.strip():
        st.error("Please enter a financial news sentence or question.")
        return

    query = user_input.strip()

    # --- Retrieval ---
    with st.spinner("Retrieving relevant context..."):
        retriever = load_retriever()
        results = retriever.retrieve(query, top_k=top_k)
        context = format_context(results)

    st.subheader("Retrieved Context")
    for i, r in enumerate(results, start=1):
        with st.expander(f"Chunk {i} — {r['source']} (score={r['score']:.3f})", expanded=(i == 1)):
            st.markdown(r["text"])

    # --- Inference ---
    if not model_ok:
        st.subheader("Sentiment & Reasoning")
        st.error("Model adapter unavailable — train V2 first to generate sentiment analysis.")
        return

    with st.spinner("Running instruction-tuned inference..."):
        model, tokenizer = load_instruction_model()
        augmented = build_augmented_input(query, context)
        response = generate_response(
            model,
            tokenizer,
            INSTRUCTION,
            augmented,
            max_new_tokens=256,
            temperature=0.0,
        )

    st.subheader("Sentiment & Reasoning")
    st.markdown(response)

    with st.expander("Pipeline details"):
        st.markdown(
            f"- **Base model:** `{BASE_MODEL}`\n"
            f"- **Adapter:** `{ADAPTER_PATH}`\n"
            f"- **Compute dtype:** `{get_compute_dtype()}`\n"
            f"- **RAG chunks retrieved:** {len(results)}"
        )


if __name__ == "__main__":
    main()
