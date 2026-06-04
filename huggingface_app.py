import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chat-message-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-ai {
        background: #1e1e2e;
        color: #e0e0e0;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid #333;
    }
    .source-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .stat-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RAG PIPELINE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Loading AI pipeline...")
def load_pipeline():
    from backend.core.rag_pipeline import RAGPipeline
    return RAGPipeline()

pipeline = load_pipeline()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 AI Knowledge Assistant")
    st.markdown("**RAG · LANGCHAIN · GROQ**")
    st.divider()

    page = st.selectbox("Navigate", [
        "🏠 Dashboard",
        "📄 Upload Documents",
        "💬 Chat Assistant",
        "🔍 Semantic Search"
    ])

    st.divider()
    status = pipeline.get_status()
    st.markdown("**SYSTEM STATUS**")
    st.markdown(f"<span class='status-online'>✅ LLM: {status['llm_model']}</span>", unsafe_allow_html=True)
    if status['vector_store_ready']:
        st.markdown("<span class='status-online'>✅ Vector Store</span>", unsafe_allow_html=True)
        st.markdown(f"**{status['total_indexed_chunks']} chunks indexed**")

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("<h1 class='main-header'>🏠 Dashboard</h1>", unsafe_allow_html=True)
    st.caption("System overview and knowledge base statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", len(status['uploaded_files']))
    with col2:
        st.metric("Chunks Indexed", status['total_indexed_chunks'])
    with col3:
        st.metric("LLM Status", "✅ Ready" if status['llm_ready'] else "❌ Off")
    with col4:
        st.metric("RAG Chain", "✅ Active" if status['rag_chain_ready'] else "⏳ Pending")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏗️ RAG Pipeline Architecture")
        st.code("""INDEXING PIPELINE
PDF Upload → PyPDF → Text Chunks
→ sentence-transformers → Embeddings
→ FAISS Vector Store (persisted)

RETRIEVAL PIPELINE
User Question → Embed Query
→ FAISS Similarity Search → Top-K Chunks

GENERATION PIPELINE
Question + Context → Prompt Template
→ LLaMA3 via Groq → Answer + Citations""")

    with col2:
        st.markdown("### 🛠️ Tech Stack")
        tech = {
            "LangChain": "RAG orchestration",
            "FAISS": "Vector database",
            "sentence-transformers": "Local embeddings",
            "LLaMA3 via Groq": "LLM generation",
            "Streamlit": "Frontend UI",
            "PyPDF": "PDF processing"
        }
        for tech_name, purpose in tech.items():
            st.markdown(f"**{tech_name}** — {purpose}")

# ─────────────────────────────────────────────
# UPLOAD DOCUMENTS
# ─────────────────────────────────────────────
elif page == "📄 Upload Documents":
    st.markdown("<h1 class='main-header'>📄 Upload Documents</h1>", unsafe_allow_html=True)
    st.caption("Add PDF documents to your AI knowledge base")

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary"):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        result = pipeline.ingest_document(tmp_path)
                        if result['success']:
                            st.success(f"✅ {result['file_name']} indexed successfully!")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Pages", result['pages_processed'])
                            col2.metric("Chunks", result['chunks_created'])
                            col3.metric("Avg Chunk", f"{result['avg_chunk_size']} ch...")
                            col4.metric("Time", f"{result['processing_time_sec']}s")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    finally:
                        os.unlink(tmp_path)

    st.divider()
    st.markdown("### How It Works")
    steps = [
        "PDF text is extracted using PyPDF",
        "Split into 1000-char chunks with 200-char overlap",
        "Each chunk embedded into a 384-dim vector",
        "Vectors stored in FAISS index on disk",
        "Ready for semantic search & Q&A!"
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**Step {i}:** {step}")

# ─────────────────────────────────────────────
# CHAT ASSISTANT
# ─────────────────────────────────────────────
elif page == "💬 Chat Assistant":
    st.markdown("<h1 class='main-header'>💬 Chat Assistant</h1>", unsafe_allow_html=True)
    st.caption("Ask questions about your uploaded documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-message-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message-ai'>{msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} Source(s)"):
                    for i, source in enumerate(msg['sources'], 1):
                        st.markdown(f"""<div class='source-card'>
                            <strong>Source {i}: {source['file']}</strong> — Page {source['page']}<br>
                            <small>{source['content']}</small>
                        </div>""", unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input("Ask a question...", placeholder="e.g., What is the main topic of this document?", label_visibility="collapsed")
    with col2:
        send = st.button("Send →", type="primary")

    if send and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("🤔 Thinking..."):
            try:
                result = pipeline.query(question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                })
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# SEMANTIC SEARCH
# ─────────────────────────────────────────────
elif page == "🔍 Semantic Search":
    st.markdown("<h1 class='main-header'>🔍 Semantic Search</h1>", unsafe_allow_html=True)
    st.caption("Search your knowledge base without LLM generation")

    query = st.text_input("Search query", placeholder="Enter search terms...")
    k = st.slider("Number of results", 1, 10, 5)

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = pipeline.semantic_search(query, k=k)
            if results:
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} — {result['file']} (Page {result['page']}) — Score: {result['score']}"):
                        st.write(result['content'])
            else:
                st.warning("No results found. Upload documents first!")