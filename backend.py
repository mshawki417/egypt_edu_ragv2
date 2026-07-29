import os
import re
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ── Lazy-import heavy deps so the page loads fast ─────────────────────────────
@st.cache_resource(show_spinner="Loading NLP libraries…")
def _load_libs():
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return True

_load_libs()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Load data ─────────────────────────────────────────────────────────────────
from technest_data import DOCUMENTS, GROUND_TRUTH, OUT_OF_SCOPE_QUERIES, is_numeric_query

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_lexical(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

# ── Build index (cached so it runs once per session) ──────────────────────────
@st.cache_resource(show_spinner="Building retrieval index…")
def build_index(documents: tuple):  # tuple so it's hashable for Streamlit cache
    docs = list(documents)  # convert back from tuple

    # Chunk
    rows = []
    for doc in docs:
        for ci, chunk in enumerate(chunk_text(doc["text"])):
            rows.append({
                "chunk_id": f"doc{doc['document_id']}_chunk{ci}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "category": doc["category"],
                "doc_type": doc["doc_type"],
                "effective_date": doc["effective_date"],
                "is_current": doc["is_current"],
                "chunk_text": chunk,
                "search_text": f"{doc['title']} {doc['category']} {doc['doc_type']} {chunk}",
            })

    chunks_df = pd.DataFrame(rows)

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(chunks_df["search_text"].map(normalize_lexical))

    # BM25
    tokenized = [simple_tokenize(t) for t in chunks_df["search_text"]]
    bm25 = BM25Okapi(tokenized)

    # Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks_df["search_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return chunks_df, tfidf, tfidf_matrix, bm25, model, embeddings

chunks_df, tfidf_vec, tfidf_mat, bm25_index, emb_model, emb_matrix = build_index(
    tuple(d.copy() for d in DOCUMENTS)
)

# ── Retrievers ────────────────────────────────────────────────────────────────
def retrieve_tfidf(query: str, k: int = 5):
    vec = tfidf_vec.transform([normalize_lexical(query)])
    scores = cosine_similarity(vec, tfidf_mat).flatten()
    idx = np.argsort(scores)[::-1][:k]
    res = chunks_df.iloc[idx].copy()
    res["score"] = scores[idx]
    res["retriever"] = "TF-IDF"
    return res.reset_index(drop=True)

def retrieve_bm25(query: str, k: int = 5):
    scores = bm25_index.get_scores(simple_tokenize(query))
    idx = np.argsort(scores)[::-1][:k]
    res = chunks_df.iloc[idx].copy()
    res["score"] = scores[idx]
    res["retriever"] = "BM25"
    return res.reset_index(drop=True)

def retrieve_semantic(query: str, k: int = 5):
    qvec = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(qvec, emb_matrix).flatten()
    idx = np.argsort(scores)[::-1][:k]
    res = chunks_df.iloc[idx].copy()
    res["score"] = scores[idx]
    res["retriever"] = "Semantic"
    return res.reset_index(drop=True)

def retrieve_hybrid(query: str, alpha: float = 0.6, k: int = 5):
    lex_vec = tfidf_vec.transform([normalize_lexical(query)])
    lex_scores = cosine_similarity(lex_vec, tfidf_mat).flatten()
    qvec = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sem_scores = cosine_similarity(qvec, emb_matrix).flatten()
    combined = alpha * min_max_normalize(sem_scores) + (1 - alpha) * min_max_normalize(lex_scores)
    idx = np.argsort(combined)[::-1][:k]
    res = chunks_df.iloc[idx].copy()
    res["tfidf_score"] = lex_scores[idx]
    res["semantic_score"] = sem_scores[idx]
    res["score"] = combined[idx]
    res["retriever"] = f"Hybrid α={alpha}"
    return res.reset_index(drop=True)

# ── Context builder ───────────────────────────────────────────────────────────
def build_context(query: str, alpha: float = 0.6, retrieval_k: int = 10,
                  max_chunks: int = 3, word_budget: int = 200,
                  min_score_ratio: float = 0.40):
    candidates = retrieve_hybrid(query, alpha=alpha, k=retrieval_k)
    candidates = candidates.sort_values(
        ["is_current", "score", "effective_date"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    max_score = candidates["score"].max() if len(candidates) else 0.0
    selected, seen, per_doc, used_words = [], set(), {}, 0

    for _, row in candidates.iterrows():
        if max_score > 0 and row["score"] < max_score * min_score_ratio:
            continue
        norm = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if norm in seen:
            continue
        if per_doc.get(row["document_id"], 0) >= 1:
            continue
        cw = len(row["chunk_text"].split())
        if selected and used_words + cw > word_budget:
            continue
        selected.append(row.to_dict())
        seen.add(norm)
        per_doc[row["document_id"]] = per_doc.get(row["document_id"], 0) + 1
        used_words += cw
        if len(selected) >= max_chunks:
            break

    blocks = []
    for i, row in enumerate(selected, 1):
        label = "CURRENT" if row["is_current"] else "OUTDATED"
        blocks.append(f"[Source {i}] {row['title']} | {row['effective_date']} | {label}\n{row['chunk_text']}")

    return {
        "context_text": "\n\n".join(blocks),
        "selected": selected,
        "used_words": used_words,
        "candidates": candidates,
    }

# ── Prompt builders ───────────────────────────────────────────────────────────
def build_strict_prompt(query: str, context: str) -> str:
    return f"""You are a grounded RAG assistant for TechNest customer support.

Rules:
1. Use only the provided context. Never add background knowledge.
2. If the answer is not in the context, say: "The provided sources do not contain enough information to answer this question."
3. If a source is marked OUTDATED, do not use it as the primary answer. Mention it only to note the conflict.
4. If current and outdated sources conflict, state the conflict and use the CURRENT source.
5. Output exactly two sections:
   Answer: [your grounded answer]
   Sources: [list the source numbers you used]

Question:
{query}

Context:
{context}
"""

def build_better_prompt(query: str, context: str) -> str:
    return f"""You are a careful TechNest customer support assistant.

Answer using only the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context is not enough to answer, say so clearly.
3. If sources disagree, prefer the most current source and mention the conflict.
4. Cite the source numbers you use in your answer.
5. Keep the answer concise but complete.

Question:
{query}

Context:
{context}
"""

PROMPT_BUILDERS = {"strict": build_strict_prompt, "better": build_better_prompt}

# ── LLM call ─────────────────────────────────────────────────────────────────
def generate_answer(query: str, context: str, prompt_style: str = "strict", model_name: str = "", api_key: str = "") -> tuple[str, bool]:
    """Returns (answer_text, used_llm)."""
    prompt = PROMPT_BUILDERS[prompt_style](query, context)
    
    if not api_key or not model_name:
        return (
            "⚠️ No API key or Model Name provided. Please configure OpenRouter settings in the sidebar.\n\n"
            "**Retrieved evidence (fallback):**\n\n" + context,
            False,
        )
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        return response.choices[0].message.content, True
    except Exception as e:
        return (f"⚠️ Error generating answer: {str(e)}\n\n**Retrieved evidence (fallback):**\n\n{context}", False)

# ── Evaluation helpers ────────────────────────────────────────────────────────
def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def hit_rate_at_k(retrieved, relevant, k):
    return int(len(set(retrieved[:k]) & set(relevant)) > 0)

def reciprocal_rank(retrieved, relevant):
    for i, d in enumerate(retrieved, 1):
        if d in set(relevant):
            return 1 / i
    return 0.0

@st.cache_data(show_spinner="Running evaluation (this takes ~30 s)…")
def run_full_eval(k: int = 3):
    rows = []
    for query, rel_ids in GROUND_TRUTH.items():
        for name, fn in [
            ("TF-IDF", lambda q, k: retrieve_tfidf(q, k)),
            ("BM25",   lambda q, k: retrieve_bm25(q, k)),
            ("Semantic", lambda q, k: retrieve_semantic(q, k)),
            ("Hybrid α=0.3", lambda q, k: retrieve_hybrid(q, 0.3, k)),
            ("Hybrid α=0.5", lambda q, k: retrieve_hybrid(q, 0.5, k)),
            ("Hybrid α=0.7", lambda q, k: retrieve_hybrid(q, 0.7, k)),
        ]:
            ret_ids = fn(query, k)["document_id"].tolist()
            rows.append({
                "retriever": name,
                "query": query,
                f"P@{k}": precision_at_k(ret_ids, rel_ids, k),
                f"R@{k}": recall_at_k(ret_ids, rel_ids, k),
                f"HR@{k}": hit_rate_at_k(ret_ids, rel_ids, k),
                "RR": reciprocal_rank(ret_ids, rel_ids),
                "is_numeric": is_numeric_query(query),
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("retriever")[[f"P@{k}", f"R@{k}", f"HR@{k}", "RR"]].mean()
    return df, summary.sort_values("RR", ascending=False)

# LLM Answer Evaluator
def evaluate_answer_llm(query: str, answer: str, context: str, model_name: str, api_key: str) -> str:
    """Evaluates the generated answer using an LLM-as-a-judge approach."""
    if not api_key or not model_name:
        return "⚠️ OpenRouter API Key and Model Name are required for evaluation."
        
    prompt = f"""You are an expert evaluator. Your task is to evaluate the quality of a generated answer for a customer support question based ONLY on the provided context.

Context:
{context}

Question:
{query}

Generated Answer:
{answer}

Please evaluate the answer on the following two criteria:
1. Groundedness (Is the answer fully supported by the context? Does it avoid outside knowledge?)
2. Relevance (Does the answer directly address the user's question?)

Provide your evaluation in the following format:
Groundedness Score: [1-5]
Relevance Score: [1-5]
Explanation: [Brief explanation of your scores]
"""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Evaluation error: {str(e)}"
