# TechNest RAG Pipeline 🛍️

A complete, production-ready **Retrieval-Augmented Generation** system for a fictional electronics store — built as a capstone project demonstrating every layer of a real RAG stack.

## Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## What's Inside

| File | Purpose |
|---|---|
| `technest_data.py` | Standalone knowledge base (40 docs) + ground truth (36 queries) |
| `app.py` | Full Streamlit app — ask, compare, evaluate, browse data |
| `technest_rag_pipeline.ipynb` | Original teaching notebook (untouched) |
| `requirements.txt` | All Python dependencies |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│               Hybrid Retriever                  │
│  ┌─────────────┐        ┌─────────────────────┐ │
│  │  TF-IDF +   │ α=0.4  │  Sentence-BERT      │ │
│  │  BM25       │◄──────►│  all-MiniLM-L6-v2   │ │
│  │  (lexical)  │        │  (semantic)          │ │
│  └─────────────┘        └─────────────────────┘ │
└─────────────────────────────────────────────────┘
    │
    ▼
Context Builder
  • Prefer is_current=True sources
  • Deduplicate chunks
  • Enforce word budget
  • Label CURRENT / OUTDATED
    │
    ▼
Prompt (strict / better)
    │
    ▼
Claude claude-sonnet-4-6 → Grounded Answer
```

---

## Key Design Decisions

### Ground Truth: 3 Query Tiers

| Tier | Description | Why |
|---|---|---|
| **A — Semantic** | Paraphrase queries ("money back" → refund doc) | Tests embedding strength |
| **B — Numeric** | Exact-number queries ("$4.99", "30 days", "500 points") | **Exposes embedding weakness** — semantic vectors blur precise numbers |
| **C — Multi-doc** | Answers spread across 2+ documents | Tests coverage and recall |

### Why Numeric Queries Break Embeddings

Sentence embeddings encode *meaning*. "4.99 USD" and "7.99 USD" have nearly identical vectors because they mean the same thing semantically (a price). BM25/TF-IDF, however, treats `4` `99` and `7` `99` as different tokens — so it correctly distinguishes the current vs. old shipping rate.

The **hybrid retriever** (α blend of semantic + lexical) handles both cases.

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/technest-rag
cd technest-rag
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

No API key? The app still runs — it shows the retrieved evidence as a fallback.

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy — done ✅

---

## Knowledge Base Stats

- **40 documents** across 12 categories
- **35 current** + **5 deliberately outdated** (to test staleness filtering)
- **36 ground-truth queries** (18 semantic + 12 numeric-detail + 6 multi-doc)
- **4 out-of-scope** queries (to test "I don't know" behavior)

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Precision@K | Of top-K results, fraction that are relevant |
| Recall@K | Of all relevant docs, fraction found in top-K |
| Hit Rate@K | At least one relevant doc in top-K? |
| MRR | Mean Reciprocal Rank — how early does the first hit appear |

Run the **Evaluation** tab in the app to see all retrievers compared.
