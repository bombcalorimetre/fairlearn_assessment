# RAG Fairness Audit — Qwen2.5 3B + Fairlearn

Audit whether your local RAG pipeline gives equally good answers
across demographic groups (gender × age group).

---

## Project Structure

```
fairlearn_rag/
├── fairlearn_rag_notebook.ipynb   ← Main notebook (run this)
├── requirements.txt
├── src/
│   ├── dataset.py        ← Medical documents + fairness test queries
│   ├── rag_pipeline.py   ← FAISS store + Qwen2.5 generator + RAG class
│   ├── fairness_eval.py  ← Scoring functions + Fairlearn MetricFrame
│   └── visualize.py      ← Dashboard charts
└── outputs/              ← Generated CSVs and PNGs
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebook
jupyter notebook fairlearn_rag_notebook.ipynb
```

---

## What Each File Does

### `src/dataset.py`
- `MEDICAL_DOCUMENTS`: 19 medical text paragraphs used as the RAG corpus
- `build_fairness_dataset()`: returns a DataFrame of 16 queries tagged
  with `gender` and `age_group` as sensitive features

### `src/rag_pipeline.py`
- `FAISSDocumentStore`: embeds docs with `all-MiniLM-L6-v2`, stores in FAISS
- `QwenGenerator`: wraps `Qwen2.5-3B-Instruct` for context-grounded generation
- `RAGPipeline`: orchestrates retrieve → generate

### `src/fairness_eval.py`
- `keyword_score()`: fraction of expected keywords found in the answer
- `refusal_score()`: detects if the model refused to answer
- `composite_score()`: binary good/poor label (combines above)
- `RAGFairnessEvaluator`: runs the pipeline + builds Fairlearn MetricFrame reports

### `src/visualize.py`
- `full_dashboard()`: 2×2 matplotlib dashboard
- Individual plot functions for scores, distributions, heatmaps, disparities

---

## Notebook Flow

| Cell | What It Does |
|------|-------------|
| 0    | Install deps |
| 1    | Imports |
| 2-3  | Build FAISS index, test retrieval |
| 4    | Load Qwen2.5-3B |
| 5    | Assemble pipeline |
| 6-7  | Load and inspect fairness dataset |
| 8    | Run evaluation (generates answers + scores) |
| 9-10 | Fairness reports for gender and age group |
| 11   | Full visual dashboard |
| 12   | Topic-level drill-down |
| 13   | Inspect raw answers |
| 14   | Mitigation: prompt engineering + ThresholdOptimizer |
| 15   | Save outputs |

---

## Interpreting Results

| Metric | Meaning |
|--------|---------|
| `keyword_score` | Float 0–1: how many expected keywords appeared in the answer |
| `refusal_score` | 1 = answered, 0 = refused |
| `composite_label` | Binary: 1 = good answer, 0 = poor answer |
| `dp_difference` | Demographic parity gap; **≤ 0.1 = fair**, > 0.1 = biased |
| `disparity` | Max accuracy gap across groups per metric |

---

## Requirements

- Python 3.10+
- ~6 GB VRAM for GPU inference, ~12 GB RAM for CPU
- Qwen2.5-3B-Instruct downloaded from HuggingFace on first run
