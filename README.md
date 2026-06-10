# Financial LLM Fine-Tuning with QLoRA

End-to-end financial sentiment instruction tuning using **Qwen2.5**, **LoRA**, **QLoRA**, and **Hugging Face Transformers**.

This project demonstrates how to build a complete LLM fine-tuning pipeline for financial sentiment analysis. Starting from [Financial PhraseBank](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank), the workflow includes dataset preparation, instruction dataset generation, parameter-efficient fine-tuning with QLoRA, and inference with natural-language reasoning.

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Method** | QLoRA (4-bit NF4 + LoRA adapters) |
| **Dataset** | Financial PhraseBank — 4,840 labeled sentences |
| **Task** | Financial sentiment classification with reasoning |
| **Stack** | PyTorch · Transformers · PEFT · TRL · bitsandbytes |
| **V3 Demo** | RAG + Streamlit financial AI assistant |

---

## Key Highlights

- End-to-end LLM fine-tuning pipeline
- Instruction tuning using Qwen2.5
- Parameter-efficient training with LoRA / QLoRA
- Automated dataset generation workflow
- Financial sentiment classification with reasoning generation
- Reproducible training and inference process
- RAG-powered demo combining retrieval with instruction-tuned inference

---

## Architecture

```
Financial PhraseBank
        ↓
download_dataset.py
        ↓
create_instruction_dataset.py
        ↓
Instruction Dataset
        ↓
Qwen2.5 + QLoRA
        ↓
LoRA Adapter
        ↓
infer_instruction.py
        ↓
Sentiment + Reasoning Output
```

| Stage | Script | Output |
|-------|--------|--------|
| Data ingestion | `download_dataset.py` | `data/train.jsonl`, `validation.jsonl`, `test.jsonl` |
| Instruction formatting | `create_instruction_dataset.py` | `data/instruction/*.jsonl` |
| Fine-tuning | `train_instruction.py` | `results/model-instruction/` |
| Inference | `infer_instruction.py` | Structured sentiment + reason |

The base model weights remain frozen in 4-bit precision; only LoRA adapter layers (~0.1–1% of parameters) are updated during training via TRL's `SFTTrainer` with assistant-only loss.

---

## Training Results

V2 instruction tuning on Google Colab (NVIDIA T4, `max_steps=20`):

![V2 QLoRA training metrics](assets/training_result.png)

| Metric | Value |
|--------|-------|
| Train loss | 0.695 |
| Eval loss | 0.304 |
| Eval mean token accuracy | **91.9%** |

Validation loss fell below training loss and token accuracy exceeded 91%, confirming the adapter learned the structured `Sentiment + Reason` output format within a short smoke-test run.

---

## Inference Demo

The instruction-tuned model generates sentiment classification **with reasoning** — not just a label:

![V2 instruction-tuned inference output](assets/inference_demo.png)

**Input:** `Apple reported record earnings.`

**Output:**
```
Sentiment: Positive

Reason:
The company exceeded earnings expectations and reported strong financial performance.
```

Each response includes a sentiment label and a short explanation grounded in the financial sentence. See [`examples/instruction_examples.md`](examples/instruction_examples.md) for additional examples.

---

## V1 vs V2

The repository ships two experimental tracks on the same dataset and base model:

| | **V1 — Classification** | **V2 — Instruction tuning** |
|---|---|---|
| Output | Single label (`positive`) | Sentiment + natural-language reason |
| Training | `train.py` | `train_instruction.py` |
| Inference | `inference.py` | `infer_instruction.py` |
| Adapter path | `results/model` | `results/model-instruction` |

V2 is the primary portfolio track — it showcases instruction tuning and explainable financial NLP rather than bare classification.

---

## V3 RAG Financial Assistant

V3 combines **retrieval-augmented generation (RAG)** with the V2 instruction-tuned model in a **Streamlit** web interface. Users submit financial news or questions; the system retrieves relevant domain context from a local knowledge base, then the fine-tuned model produces structured sentiment analysis with natural-language reasoning.

### Architecture

```
User Query
     ↓
Sentence Transformer Embedding
     ↓
FAISS Retrieval
     ↓
Knowledge Base Context
     ↓
Instruction-Tuned Financial Model
     ↓
Sentiment + Reasoning Output
```

| Layer | Implementation |
|-------|----------------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (`rag_index/`) |
| Knowledge base | `data/knowledge_base/*.md` |
| Generator | `Qwen/Qwen2.5-0.5B-Instruct` + V2 LoRA adapter |
| Interface | `app.py` (Streamlit) |

### Features

- Retrieval-Augmented Generation (RAG)
- FAISS Vector Search
- Financial Knowledge Base
- Streamlit Web Interface
- Financial Sentiment Reasoning

| Script | Role |
|--------|------|
| `build_rag_index.py` | Chunk markdown, embed, build FAISS index |
| `rag_retrieve.py` | Top-k semantic retrieval for a user query |
| `app.py` | Streamlit UI — context, sentiment, and reasoning |

### Run V3

```bash
# 1. Build RAG index (one-time, or after editing knowledge base)
python src/build_rag_index.py

# 2. Train V2 adapter (if not already done)
python src/download_dataset.py
python src/create_instruction_dataset.py
python src/train_instruction.py

# 3. Launch Streamlit demo
streamlit run app.py
```

**Colab one-liner setup:**

```bash
pip install -q -r requirements.txt
python src/build_rag_index.py
streamlit run app.py
```

> `rag_index/`, model adapters, and caches are **not committed** — rebuild locally or in Colab.

---

## Reproducing Results

Clone the repo on a GPU runtime (Google Colab / Kaggle). No data or model weights are committed — the full pipeline regenerates everything:

```python
%cd /content
!git clone https://github.com/YALINYAN-YU/financial-llm-finetuning.git
%cd financial-llm-finetuning

!pip install -q -r requirements-train.txt

!python src/download_dataset.py
!python src/create_instruction_dataset.py
!python src/train_instruction.py
!python src/infer_instruction.py
```

Each script validates its inputs and prints the next step on success.

---

## Project Structure

```
financial-llm-finetuning/
├── README.md
├── app.py                           # V3 Streamlit demo
├── requirements.txt
├── requirements-train.txt
├── requirements-rag.txt             # V3 RAG + Streamlit deps
├── assets/
├── data/
│   └── knowledge_base/              # V3 curated docs (committed)
├── examples/
├── src/
│   ├── download_dataset.py
│   ├── create_instruction_dataset.py
│   ├── train_instruction.py
│   ├── infer_instruction.py
│   ├── build_rag_index.py           # V3 index builder
│   ├── rag_retrieve.py              # V3 retrieval
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── rag_index/                       # Generated — not committed
└── results/                         # Adapters — not committed
```

---

## V1 Experiment (Baseline)

Early smoke test on single-label classification (`train.py`):

| Metric | Value |
|--------|-------|
| Train loss | 0.3853 |
| Validation loss | 0.1906 |
| Validation token accuracy | 92.67% |

---

## License

Research and portfolio use. Verify license terms for Qwen2.5 and Financial PhraseBank before commercial deployment.


## Resume Highlights

- Built an end-to-end financial LLM fine-tuning pipeline using Qwen2.5, LoRA, and QLoRA.
- Converted Financial PhraseBank into instruction-following data for sentiment reasoning.
- Developed a Retrieval-Augmented Generation (RAG) financial assistant using FAISS and sentence-transformers.
- Built a Streamlit application for interactive financial sentiment analysis and knowledge retrieval.




## Future Improvements

- SEC filing analysis
- Earnings call transcript summarization
- Multi-document retrieval
- Agent-based financial research workflow
- Evaluation with financial benchmarks