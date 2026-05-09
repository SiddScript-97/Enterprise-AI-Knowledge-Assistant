# frontend/app.py
# ============================================================
# Streamlit Frontend — Enterprise AI Knowledge Assistant
#
# Pages:
#   🏠 Home/Dashboard  → System status metrics
#   📄 Upload Docs     → Upload PDFs to knowledge base
#   💬 Chat            → Conversational Q&A with memory
#   🔍 Search          → Pure semantic search explorer
#   ℹ️  About          → Architecture overview
#
# Run: streamlit run frontend/app.py
# ============================================================

import streamlit as st
import time
from datetime import datetime

from frontend.api_client import get_client

# ─────────────────────────────────────────────
# PAGE CONFIGURATION (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Clean, modern dark theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --primary: #6C63FF;
        --primary-light: #8B85FF;
        --accent: #00D4AA;
        --bg-dark: #0F1117;
        --bg-card: #1A1D27;
        --bg-card2: #20243A;
        --text-primary: #E8E9F3;
        --text-secondary: #9B9DB5;
        --border: #2A2D3E;
        --success: #00D4AA;
        --warning: #FFB347;
        --danger: #FF6B6B;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: #0F1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #13151F;
        border-right: 1px solid #2A2D3E;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #E8E9F3;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        border-radius: 12px;
        padding: 16px 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #9B9DB5 !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #E8E9F3 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #8B85FF);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #8B85FF, #6C63FF);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
    }

    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        color: #E8E9F3;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6C63FF;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2);
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: #1A1D27;
        border: 2px dashed #2A2D3E;
        border-radius: 12px;
        color: #9B9DB5;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #6C63FF, #8B85FF);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.3);
    }

    .ai-bubble {
        background: #1A1D27;
        color: #E8E9F3;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid #2A2D3E;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .ai-label {
        font-size: 0.75rem;
        color: #6C63FF;
        font-weight: 600;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
    }

    .user-label {
        font-size: 0.75rem;
        color: #00D4AA;
        font-weight: 600;
        margin-bottom: 6px;
        text-align: right;
        letter-spacing: 0.05em;
    }

    /* Source citation cards */
    .source-card {
        background: #20243A;
        border: 1px solid #2A2D3E;
        border-left: 3px solid #6C63FF;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #9B9DB5;
    }

    .source-card strong {
        color: #E8E9F3;
    }

    /* Status badges */
    .badge-success {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-danger {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E8E9F3;
        margin-bottom: 4px;
    }

    .section-sub {
        font-size: 0.9rem;
        color: #9B9DB5;
        margin-bottom: 20px;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #2A2D3E;
        margin: 20px 0;
    }

    /* Search result cards */
    .search-card {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0;
        transition: border-color 0.2s;
    }

    .search-card:hover {
        border-color: #6C63FF;
    }

    .score-bar {
        height: 4px;
        background: linear-gradient(90deg, #6C63FF, #00D4AA);
        border-radius: 2px;
        margin-top: 8px;
    }

    /* Logo area */
    .logo-area {
        text-align: center;
        padding: 20px 0 10px;
    }

    .logo-icon {
        font-size: 2.5rem;
        display: block;
    }

    .logo-title {
        font-size: 1rem;
        font-weight: 700;
        color: #E8E9F3;
        margin-top: 6px;
        letter-spacing: -0.02em;
    }

    .logo-sub {
        font-size: 0.72rem;
        color: #6C63FF;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        color: #E8E9F3;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "authenticated": False,
        "username": "",
        "chat_history": [],       # list of {"role": "user"/"ai", "content": ..., "sources": [...]}
        "current_page": "🏠 Dashboard",
        "backend_online": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()
client = get_client()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-area">
            <span class="logo-icon">🧠</span>
            <div class="logo-title">AI Knowledge Assistant</div>
            <div class="logo-sub">RAG · LangChain · Gemini</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Navigation
        pages = [
            "🏠 Dashboard",
            "📄 Upload Documents",
            "💬 Chat Assistant",
            "🔍 Semantic Search",
            "ℹ️ About",
        ]

        st.session_state.current_page = st.selectbox(
            "Navigate",
            pages,
            index=pages.index(st.session_state.current_page),
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Backend status indicator
        is_online = client.health_check()
        st.session_state.backend_online = is_online

        if is_online:
            st.markdown('<span class="badge-success">● Backend Online</span>', unsafe_allow_html=True)
            status_data, err = client.get_status()
            if status_data:
                st.markdown(f"""
                <div style="margin-top:12px; padding:10px; background:#1A1D27; border-radius:8px; border:1px solid #2A2D3E;">
                    <div style="font-size:0.75rem; color:#9B9DB5; margin-bottom:8px;">SYSTEM STATUS</div>
                    <div style="font-size:0.82rem; color:#E8E9F3; margin:4px 0;">
                        {'✅' if status_data.get('llm_ready') else '❌'} LLM: {status_data.get('llm_model', 'N/A')}
                    </div>
                    <div style="font-size:0.82rem; color:#E8E9F3; margin:4px 0;">
                        {'✅' if status_data.get('vector_store_ready') else '⏳'} Vector Store
                    </div>
                    <div style="font-size:0.82rem; color:#6C63FF; margin-top:6px; font-weight:600;">
                        {status_data.get('total_indexed_chunks', 0):,} chunks indexed
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-danger">● Backend Offline</span>', unsafe_allow_html=True)
            st.caption("Start backend: `uvicorn backend.main:app --reload`")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Auth status
        if st.session_state.authenticated:
            st.markdown(f'<div style="font-size:0.82rem; color:#9B9DB5;">Logged in as <strong style="color:#E8E9F3;">{st.session_state.username}</strong></div>', unsafe_allow_html=True)
            if st.button("Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.rerun()


# ─────────────────────────────────────────────
# PAGE: LOGIN
# ─────────────────────────────────────────────

def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:40px 0 20px;">
            <div style="font-size:4rem;">🧠</div>
            <h1 style="color:#E8E9F3; font-weight:700; margin:10px 0 4px;">AI Knowledge Assistant</h1>
            <p style="color:#9B9DB5; font-size:1rem;">Sign in to access your knowledge base</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            if st.button("Sign In →", use_container_width=True):
                if not client.health_check():
                    st.error("⚠️ Backend is offline. Please start the FastAPI server first.")
                else:
                    data, err = client.login(username, password)
                    if data:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Login successful! Redirecting...")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"Login failed: {err}")

        st.markdown("""
        <div style="text-align:center; margin-top:16px; font-size:0.8rem; color:#9B9DB5;">
            Default credentials: admin / admin123<br>
            Set in .env file: AUTH_USERNAME / AUTH_PASSWORD
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

def page_dashboard():
    st.markdown('<div class="section-header">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">System overview and knowledge base statistics</div>', unsafe_allow_html=True)

    if not st.session_state.backend_online:
        st.error("⚠️ Backend is offline. Please run: `uvicorn backend.main:app --reload`")
        st.code("cd enterprise-ai-assistant\nuvicorn backend.main:app --reload", language="bash")
        return

    status_data, err = client.get_status()
    if err:
        st.error(f"Error fetching status: {err}")
        return

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        docs = len(status_data.get("uploaded_files", []))
        st.metric("📄 Documents", docs, help="Total PDFs uploaded")
    with col2:
        chunks = status_data.get("total_indexed_chunks", 0)
        st.metric("🧩 Chunks Indexed", f"{chunks:,}", help="Total text chunks in FAISS")
    with col3:
        llm = "✅ Ready" if status_data.get("llm_ready") else "❌ No Key"
        st.metric("🤖 LLM Status", llm, help="Gemini API connection status")
    with col4:
        rag = "✅ Active" if status_data.get("rag_chain_ready") else "⏳ Waiting"
        st.metric("🔗 RAG Chain", rag, help="RAG pipeline readiness")

    st.markdown("---")

    # Architecture overview
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### 🏗️ RAG Pipeline Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────┐
        │           INDEXING PIPELINE                  │
        │  PDF Upload → PyPDF → Text Chunks            │
        │  → sentence-transformers → Embeddings        │
        │  → FAISS Vector Store (persisted to disk)    │
        └─────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │           RETRIEVAL PIPELINE                 │
        │  User Question → Embed Query                 │
        │  → FAISS Similarity Search → Top-K Chunks   │
        └─────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │           GENERATION PIPELINE                │
        │  Question + Context → Prompt Template        │
        │  → Gemini 1.5 Flash → Answer + Citations    │
        └─────────────────────────────────────────────┘
        ```
        """)

    with col_right:
        st.markdown("#### 📁 Tech Stack")
        stack = {
            "LangChain": "RAG orchestration",
            "FAISS": "Vector database",
            "sentence-transformers": "Local embeddings",
            "Gemini 1.5 Flash": "LLM generation",
            "FastAPI": "REST API backend",
            "Streamlit": "Frontend UI",
            "PyPDF": "PDF processing",
        }
        for tech, desc in stack.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid #2A2D3E;">
                <span style="color:#E8E9F3;font-weight:500;font-size:0.88rem;">{tech}</span>
                <span style="color:#9B9DB5;font-size:0.8rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Uploaded documents list
    st.markdown("#### 📚 Knowledge Base Documents")
    uploaded = status_data.get("uploaded_files", [])
    if not uploaded:
        st.info("📭 No documents uploaded yet. Go to **Upload Documents** to add PDFs.")
    else:
        for doc in uploaded:
            with st.expander(f"📄 {doc['name']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Pages", doc.get("pages", "N/A"))
                c2.metric("Chunks", doc.get("chunks", "N/A"))
                c3.metric("Path", "data/uploads/")


# ─────────────────────────────────────────────
# PAGE: UPLOAD DOCUMENTS
# ─────────────────────────────────────────────

def page_upload():
    st.markdown('<div class="section-header">📄 Upload Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Add PDF documents to your AI knowledge base</div>', unsafe_allow_html=True)

    if not st.session_state.backend_online:
        st.error("⚠️ Backend is offline.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files. Max 50MB each.",
        )

        if uploaded_files:
            if st.button("🚀 Process & Index Documents"):
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result, err = client.upload_document(
                            uploaded_file.read(),
                            uploaded_file.name,
                        )

                    if result and result.get("success"):
                        st.success(f"✅ **{uploaded_file.name}** indexed successfully!")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Pages", result.get("pages_processed"))
                        col_b.metric("Chunks", result.get("chunks_created"))
                        col_c.metric("Avg Chunk", f"{result.get('avg_chunk_size', 0)} chars")
                        col_d.metric("Time", f"{result.get('processing_time_sec', 0)}s")
                    else:
                        st.error(f"❌ Failed to upload {uploaded_file.name}: {err}")

    with col2:
        st.markdown("""
        <div style="background:#1A1D27;border:1px solid #2A2D3E;border-radius:10px;padding:16px;">
            <div style="color:#6C63FF;font-size:0.8rem;font-weight:700;letter-spacing:0.05em;margin-bottom:12px;">
                HOW IT WORKS
            </div>
            <div style="font-size:0.85rem;color:#9B9DB5;line-height:1.8;">
                <b style="color:#E8E9F3;">Step 1:</b> PDF text is extracted using PyPDF<br>
                <b style="color:#E8E9F3;">Step 2:</b> Split into 1000-char chunks with 200-char overlap<br>
                <b style="color:#E8E9F3;">Step 3:</b> Each chunk is embedded into a 384-dim vector<br>
                <b style="color:#E8E9F3;">Step 4:</b> Vectors stored in FAISS index on disk<br>
                <b style="color:#E8E9F3;">Step 5:</b> Ready for semantic search & Q&A!
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🗑️ Clear All Documents", help="Remove all documents from knowledge base"):
            with st.spinner("Clearing..."):
                result, err = client.clear_documents()
            if result:
                st.success("Knowledge base cleared!")
                st.rerun()
            else:
                st.error(f"Error: {err}")


# ─────────────────────────────────────────────
# PAGE: CHAT ASSISTANT
# ─────────────────────────────────────────────

def page_chat():
    st.markdown('<div class="section-header">💬 Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask questions about your uploaded documents</div>', unsafe_allow_html=True)

    if not st.session_state.backend_online:
        st.error("⚠️ Backend is offline.")
        return

    # Chat controls
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Render chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#9B9DB5;">
                <div style="font-size:3rem;margin-bottom:12px;">💬</div>
                <div style="font-size:1.1rem;font-weight:600;color:#E8E9F3;">Start a conversation</div>
                <div style="font-size:0.9rem;margin-top:8px;">
                    Upload documents first, then ask questions about them.<br>
                    The AI will answer with citations from your documents.
                </div>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-label">YOU</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-label">🧠 AI ASSISTANT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

                # Show source citations
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 {len(sources)} Source(s)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}: {src.get('file', 'Unknown')}</strong>
                                — Page {src.get('page', '?')}<br>
                                <span style="font-size:0.8rem;color:#6B6E8A;">
                                {src.get('content', '')[:200]}...
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g., What are the main findings of the report?",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question.strip(),
        })

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            result, err = client.chat(question.strip())

        if result:
            st.session_state.chat_history.append({
                "role": "ai",
                "content": result.get("answer", "No answer generated."),
                "sources": result.get("sources", []),
                "time": result.get("processing_time", 0),
            })
        else:
            st.session_state.chat_history.append({
                "role": "ai",
                "content": f"⚠️ Error: {err}",
                "sources": [],
            })

        st.rerun()

    # Sample questions
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        sample_qs = [
            "What is the main topic of the document?",
            "Summarize the key findings",
            "What are the conclusions?",
            "List the main recommendations",
        ]
        cols = st.columns(2)
        for i, q in enumerate(sample_qs):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        result, err = client.chat(q)
                    if result:
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "content": result.get("answer", "No answer."),
                            "sources": result.get("sources", []),
                        })
                    st.rerun()


# ─────────────────────────────────────────────
# PAGE: SEMANTIC SEARCH
# ─────────────────────────────────────────────

def page_search():
    st.markdown('<div class="section-header">🔍 Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search document chunks by meaning, not just keywords</div>', unsafe_allow_html=True)

    if not st.session_state.backend_online:
        st.error("⚠️ Backend is offline.")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g., financial performance metrics")
    with col2:
        top_k = st.selectbox("Results", [3, 5, 8, 10], index=1)

    if st.button("🔍 Search"):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching..."):
            result, err = client.semantic_search(query.strip(), top_k=top_k)

        if err:
            st.error(f"Search error: {err}")
            return

        results = result.get("results", [])
        total = result.get("total_results", 0)

        st.markdown(f"**Found {total} results for:** *{query}*")
        st.markdown("---")

        if not results:
            st.info("No results found. Upload some documents first.")
            return

        for i, res in enumerate(results, 1):
            score = res.get("score", 0)
            relevance_pct = max(0, min(100, int((1 - score) * 100)))  # lower L2 = more relevant

            st.markdown(f"""
            <div class="search-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="color:#6C63FF;font-weight:600;font-size:0.85rem;">Result #{i}</span>
                    <span style="color:#9B9DB5;font-size:0.8rem;">
                        📄 {res.get('file','?')} · Page {res.get('page','?')}
                    </span>
                </div>
                <div style="color:#E8E9F3;font-size:0.88rem;line-height:1.6;">
                    {res.get('content','')[:400]}...
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
                    <div style="font-size:0.75rem;color:#9B9DB5;">Relevance:</div>
                    <div style="flex:1;height:4px;background:#2A2D3E;border-radius:2px;">
                        <div style="width:{relevance_pct}%;height:100%;background:linear-gradient(90deg,#6C63FF,#00D4AA);border-radius:2px;"></div>
                    </div>
                    <div style="font-size:0.75rem;color:#6C63FF;font-weight:600;">{relevance_pct}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT / ARCHITECTURE
# ─────────────────────────────────────────────

def page_about():
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Architecture, tech stack, and interview talking points</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🧠 Enterprise AI Knowledge Assistant

    A **RAG (Retrieval-Augmented Generation)** powered document Q&A system built with
    modern AI engineering practices.

    ---

    ### 🏗️ Architecture Overview

    ```
    User (Streamlit UI)
          │
          ▼
    FastAPI REST API  ←→  Authentication
          │
          ├──── Document Ingestion
          │         PyPDF → Text Splitter → Embeddings → FAISS
          │
          ├──── RAG Q&A
          │         Query → FAISS Retrieval → LangChain Chain → Gemini LLM
          │
          └──── Semantic Search
                    Query Embedding → FAISS ANN Search → Ranked Results
    ```

    ---

    ### 💡 Interview Talking Points

    **"What is RAG and why did you use it?"**
    > RAG (Retrieval-Augmented Generation) grounds LLM responses in real documents.
    Instead of relying on the LLM's training data, we retrieve relevant chunks from
    the document first, then pass them as context to the LLM. This prevents hallucination
    and enables domain-specific Q&A without fine-tuning.

    **"Why FAISS instead of a cloud vector DB?"**
    > FAISS is a local vector store by Facebook AI — perfect for this project because
    it's free, fast, and requires no external service. For production at scale,
    I'd migrate to Pinecone or Weaviate.

    **"Why sentence-transformers for embeddings?"**
    > all-MiniLM-L6-v2 runs locally (no API cost), produces 384-dim vectors, and is
    fast on CPU. For production, I'd evaluate OpenAI Ada-002 or Gemini embeddings.

    **"How does the chat memory work?"**
    > LangChain's ConversationBufferWindowMemory stores the last 5 Q&A pairs. This
    context is injected into each new prompt, making the assistant context-aware.

    **"How do source citations work?"**
    > LangChain's ConversationalRetrievalChain returns source_documents alongside
    the answer. I extract the file name, page number, and a snippet from each
    retrieved chunk to display as citations.

    ---

    ### 🛠️ Tech Stack
    | Component | Technology | Why |
    |-----------|-----------|-----|
    | LLM | Google Gemini 1.5 Flash | Fast, cost-effective, free tier |
    | RAG Framework | LangChain | Industry standard, interview-recognized |
    | Vector DB | FAISS | Local, fast, no external service |
    | Embeddings | sentence-transformers | Free, runs locally |
    | Backend | FastAPI | Modern, async, auto-docs |
    | Frontend | Streamlit | Rapid AI app development |
    | PDF Processing | PyPDF | Simple, reliable |

    ---

    ### 📁 Project Structure
    ```
    enterprise-ai-assistant/
    ├── backend/
    │   ├── main.py              # FastAPI app, all endpoints
    │   ├── config.py            # Settings via pydantic-settings
    │   └── core/
    │       └── rag_pipeline.py  # DocumentProcessor, VectorStore, RAG chain
    ├── frontend/
    │   ├── app.py               # Streamlit UI (5 pages)
    │   └── api_client.py        # HTTP client for FastAPI
    ├── data/                    # Auto-created
    │   ├── uploads/             # Uploaded PDFs
    │   └── vectorstore/         # FAISS index files
    ├── .env.example             # Environment variable template
    ├── requirements.txt         # All dependencies
    └── README.md                # Documentation
    ```
    """)


# ─────────────────────────────────────────────
# MAIN APP ROUTER
# ─────────────────────────────────────────────

def main():
    render_sidebar()

    # Route to pages
    page = st.session_state.current_page

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "📄 Upload Documents":
        page_upload()
    elif page == "💬 Chat Assistant":
        page_chat()
    elif page == "🔍 Semantic Search":
        page_search()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
