"""
TechNest RAG Pipeline — Streamlit App
======================================
Run locally:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNest RAG Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import logic from backend
from backend import (
    DOCUMENTS, GROUND_TRUTH, OUT_OF_SCOPE_QUERIES, is_numeric_query, chunks_df,
    retrieve_tfidf, retrieve_bm25, retrieve_semantic, retrieve_hybrid,
    build_context, generate_answer, run_full_eval, evaluate_answer_llm
)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")


    # Load API Key and Model from Streamlit Secrets
    try:
        openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
        openrouter_model = st.secrets["OPENROUTER_MODEL"]
    except Exception:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        openrouter_model = os.getenv("OPENROUTER_MODEL", "")

    st.divider()

    alpha = st.slider(
        "Hybrid alpha (semantic weight)",
        0.0, 1.0, 0.6, 0.05,
        help="1.0 = pure semantic, 0.0 = pure lexical"
    )

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
                user_query,
                pkg["context_text"],
                prompt_style=prompt_style,
            )

        st.subheader("Answer")
        st.markdown(answer)
        
        # Save state for evaluation
        st.session_state["last_query"] = user_query
        st.session_state["last_answer"] = answer
        st.session_state["last_context"] = pkg["context_text"]

        with st.expander(f"📄 Context used ({pkg['used_words']} words, {len(pkg['selected'])} sources)"):
            if pkg["context_text"]:
                st.code(pkg["context_text"], language="")
            else:
                st.warning("No sources passed the relevance filters.")

        with st.expander("🗂️ All retrieved candidates (before filtering)"):
            cols_show = ["chunk_id", "title", "is_current", "effective_date", "score", "chunk_text"]
            st.dataframe(pkg["candidates"][cols_show], use_container_width=True)

    # Add evaluation block
    if "last_answer" in st.session_state:
        st.divider()
        st.subheader("🧪 Evaluate Generated Answer")
        if st.button("Run LLM-as-a-Judge Evaluation"):
            with st.spinner("Evaluating answer using OpenRouter LLM..."):
                eval_result = evaluate_answer_llm(
                    st.session_state["last_query"],
                    st.session_state["last_answer"],
                    st.session_state["last_context"],
                
                )
            st.markdown(f"```text\n{eval_result}\n```")

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
