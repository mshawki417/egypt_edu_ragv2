"""
TechNest RAG Pipeline — Streamlit App
======================================
Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Connect at share.streamlit.io
    3. Add ANTHROPIC_API_KEY in Settings > Secrets
"""

import os
import re
import json
import string
from pathlib import Path

import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNest RAG Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    import pandas as pd
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

    import pandas as pd
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
def generate_answer(query: str, context: str, prompt_style: str = "strict") -> tuple[str, bool]:
    """Returns (answer_text, used_llm)."""
    prompt = PROMPT_BUILDERS[prompt_style](query, context)
    api_key = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return (
            "⚠️ No API key found.  Add `ANTHROPIC_API_KEY` to your environment or "
            "Streamlit secrets to enable LLM generation.\n\n"
            "**Retrieved evidence (fallback):**\n\n" + context,
            False,
        )
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text"), True

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
    import pandas as pd
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

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    alpha = st.slider("Hybrid alpha (semantic weight)", 0.0, 1.0, 0.6, 0.05,
                      help="1.0 = pure semantic, 0.0 = pure lexical")
    retrieval_k = st.slider("Retrieval pool size", 5, 20, 10)
    max_chunks = st.slider("Max context chunks", 1, 5, 3)
    word_budget = st.slider("Word budget", 100, 400, 200)
    prompt_style = st.selectbox("Prompt style", ["strict", "better"])
    st.divider()
    st.markdown(
        "**Knowledge base:** %d documents (%d current, %d outdated)"
        % (
            len(DOCUMENTS),
            sum(1 for d in DOCUMENTS if d["is_current"]),
            sum(1 for d in DOCUMENTS if not d["is_current"]),
        )
    )
    st.markdown("**Ground truth queries:** %d" % len(GROUND_TRUTH))

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_ask, tab_compare, tab_eval, tab_data = st.tabs(
    ["💬 Ask", "🔬 Retriever Comparison", "📊 Evaluation", "📂 Data"]
)

# ── Tab 1: Ask ────────────────────────────────────────────────────────────────
with tab_ask:
    st.header("Ask TechNest Assistant")

    example_queries = (
        ["— pick an example —"]
        + list(GROUND_TRUTH.keys())
        + OUT_OF_SCOPE_QUERIES
    )
    chosen = st.selectbox("Or pick an example query:", example_queries, key="ex")
    user_query = st.text_input(
        "Your question:",
        value="" if chosen.startswith("—") else chosen,
        placeholder="e.g. How much does standard shipping cost?",
    )

    if st.button("🔍 Get Answer", type="primary") and user_query.strip():
        with st.spinner("Retrieving and generating…"):
            pkg = build_context(
                user_query, alpha=alpha, retrieval_k=retrieval_k,
                max_chunks=max_chunks, word_budget=word_budget,
            )
            answer, used_llm = generate_answer(
                user_query, pkg["context_text"], prompt_style=prompt_style
            )

        st.subheader("Answer")
        st.markdown(answer)

        with st.expander(f"📄 Context used ({pkg['used_words']} words, {len(pkg['selected'])} sources)"):
            if pkg["context_text"]:
                st.code(pkg["context_text"], language="")
            else:
                st.warning("No sources passed the relevance filters.")

        with st.expander("🗂️ All retrieved candidates (before filtering)"):
            cols_show = ["chunk_id", "title", "is_current", "effective_date", "score", "chunk_text"]
            st.dataframe(pkg["candidates"][cols_show], use_container_width=True)

# ── Tab 2: Retriever Comparison ───────────────────────────────────────────────
with tab_compare:
    st.header("Compare Retrievers Side-by-Side")
    cmp_query = st.text_input(
        "Query to compare:",
        value="How much does standard shipping cost right now?",
        key="cmp",
    )
    k_cmp = st.slider("Top-K", 1, 10, 5, key="k_cmp")

    if st.button("Compare", key="cmp_btn") and cmp_query.strip():
        c1, c2, c3, c4 = st.columns(4)
        cols_out = ["document_id", "title", "is_current", "score", "chunk_text"]

        with c1:
            st.subheader("TF-IDF")
            st.dataframe(retrieve_tfidf(cmp_query, k_cmp)[cols_out], use_container_width=True)
        with c2:
            st.subheader("BM25")
            st.dataframe(retrieve_bm25(cmp_query, k_cmp)[cols_out], use_container_width=True)
        with c3:
            st.subheader("Semantic")
            st.dataframe(retrieve_semantic(cmp_query, k_cmp)[cols_out], use_container_width=True)
        with c4:
            st.subheader(f"Hybrid α={alpha}")
            st.dataframe(
                retrieve_hybrid(cmp_query, alpha, k_cmp)[
                    ["document_id", "title", "is_current", "tfidf_score", "semantic_score", "score", "chunk_text"]
                ],
                use_container_width=True,
            )

    st.info(
        "**Tip:** Try a numeric query like *'How much does standard shipping cost right now?'* "
        "and observe how the old $7.99 document (doc 21, outdated) appears in BM25 vs Hybrid."
    )

# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.header("Full Pipeline Evaluation")

    k_eval = st.slider("Top-K for evaluation", 1, 10, 3, key="k_eval")

    if st.button("Run Evaluation", type="primary", key="eval_btn"):
        detail_df, summary_df = run_full_eval(k=k_eval)

        st.subheader("Summary (all queries)")
        st.dataframe(summary_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        # Numeric vs non-numeric breakdown
        st.subheader("Numeric-detail queries vs semantic queries")
        for rname in ["BM25", "Semantic", f"Hybrid α=0.7"]:
            sub = detail_df[detail_df["retriever"] == rname]
            num = sub[sub["is_numeric"]]
            nnum = sub[~sub["is_numeric"]]
            c1, c2 = st.columns(2)
            c1.metric(f"{rname} — RR on numeric", f"{num['RR'].mean():.3f}")
            c2.metric(f"{rname} — RR on semantic", f"{nnum['RR'].mean():.3f}")

        st.subheader("Query-level detail")
        st.dataframe(detail_df, use_container_width=True)

        # Missed queries
        missed = detail_df[
            (detail_df["retriever"] == f"Hybrid α=0.7") &
            (detail_df[f"HR@{k_eval}"] == 0)
        ]
        if not missed.empty:
            st.warning(f"{len(missed)} queries missed by Hybrid α=0.7 at top-{k_eval}:")
            st.dataframe(missed[["query", f"HR@{k_eval}", "RR"]], use_container_width=True)
        else:
            st.success(f"✅ Hybrid α=0.7 hit all queries at top-{k_eval}!")

# ── Tab 4: Data ───────────────────────────────────────────────────────────────
with tab_data:
    st.header("Knowledge Base")
    import pandas as pd
    docs_df = pd.DataFrame(DOCUMENTS)
    current_filter = st.multiselect("Filter by is_current", [True, False], default=[True, False])
    cat_filter = st.multiselect("Filter by category", sorted(docs_df["category"].unique()))
    filtered = docs_df[docs_df["is_current"].isin(current_filter)]
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    st.dataframe(filtered[["document_id", "title", "category", "doc_type", "effective_date", "is_current", "text"]],
                 use_container_width=True)

    st.header("Ground Truth Queries")
    gt_rows = [{"query": q, "relevant_ids": str(ids), "is_numeric": is_numeric_query(q)}
               for q, ids in GROUND_TRUTH.items()]
    gt_df = pd.DataFrame(gt_rows)
    show_numeric = st.checkbox("Show only numeric-detail queries")
    if show_numeric:
        gt_df = gt_df[gt_df["is_numeric"]]
    st.dataframe(gt_df, use_container_width=True)

    st.header("Chunks")
    st.write(f"Total chunks: **{len(chunks_df)}** from {len(DOCUMENTS)} documents")
    st.dataframe(chunks_df[["chunk_id", "title", "is_current", "chunk_text"]].head(30),
                 use_container_width=True)
