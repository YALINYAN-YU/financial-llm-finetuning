# V2 Instruction Examples

Financial sentiment instruction-tuning examples used in this project. Each record pairs a financial news sentence with a structured response containing **Sentiment** and **Reason**.

These examples are also available as JSONL in [`instruction_examples.jsonl`](instruction_examples.jsonl) and are used by `src/infer_instruction.py` for demo inference.

---

## Instruction template

```
Analyze the sentiment of the following financial news sentence.
Classify it as negative, neutral, or positive, and explain your reasoning.
```

---

## Example 1 — Positive

**Input**

```
Apple reported record earnings.
```

**Output**

```
Sentiment: Positive

Reason:
The company exceeded earnings expectations and reported strong financial performance.
```

---

## Example 2 — Negative

**Input**

```
Sales by Seppala diminished by 6 per cent.
```

**Output**

```
Sentiment: Negative

Reason:
The language points to weakening results, with declining figures or unfavorable business developments.
```

---

## Example 3 — Neutral

**Input**

```
The purchase sum is about EUR 10mn (US$ 12.97 mn).
```

**Output**

```
Sentiment: Neutral

Reason:
The text presents factual financial information without clear positive or negative market implications.
```

---

## Example 4 — Positive

**Input**

```
Operating profit was EUR 139.7 mn, up 23% from EUR 113.8 mn.
```

**Output**

```
Sentiment: Positive

Reason:
The text signals favorable financial outcomes, with rising profits, earnings, or sales indicating positive momentum.
```

---

## Example 5 — Negative

**Input**

```
The company warned that full-year guidance may be revised downward.
```

**Output**

```
Sentiment: Negative

Reason:
The statement suggests negative implications for the company's financial health or investor sentiment.
```

---

## JSONL schema (V2)

Each line in `data/instruction/*.jsonl` follows this schema:

| Field | Description |
|-------|-------------|
| `instruction` | Task prompt |
| `input` | Financial news sentence |
| `output` | Structured response (`Sentiment` + `Reason`) |
| `sentence` | Raw sentence (same as `input`) |
| `label` | Integer label (0=negative, 1=neutral, 2=positive) |
| `sentiment` | String label |
