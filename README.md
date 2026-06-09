# financial-llm-finetuning

Fine-tune a language model on financial-domain data using Hugging Face Transformers.

## Project structure

```
financial-llm-finetuning/
├── README.md
├── requirements.txt
├── data/              # Raw and processed datasets
├── notebooks/         # Exploratory analysis and experiments
├── src/
│   ├── train.py       # Fine-tuning script
│   ├── inference.py   # Run inference with a trained model
│   └── evaluate.py    # Evaluate model performance
└── results/           # Checkpoints, logs, and evaluation outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py \
  --model-name meta-llama/Llama-3.2-1B \
  --train-file data/train.jsonl \
  --output-dir results/checkpoints
```

### Inference

```bash
python src/inference.py \
  --model-path results/checkpoints \
  --prompt "Summarize the key risks in this earnings report."
```

### Evaluation

```bash
python src/evaluate.py \
  --model-path results/checkpoints \
  --eval-file data/eval.jsonl \
  --output-dir results/eval
```

## Data format

Training and evaluation files should be JSONL with one record per line:

```json
{"instruction": "What is EBITDA?", "input": "", "output": "EBITDA stands for ..."}
{"instruction": "Analyze this filing", "input": "Revenue grew 12%...", "output": "The company shows..."}
```
