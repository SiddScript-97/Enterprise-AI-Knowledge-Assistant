# 🧠 Enterprise AI Knowledge Assistant

> A production-style **RAG (Retrieval-Augmented Generation)** powered document Q&A system built with LangChain, FAISS, Google Gemini, FastAPI, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-yellow.svg)](https://langchain.com/)

---

## 🎯 What This Project Does

Upload any PDF document and ask questions about it in natural language. The AI:
- **Retrieves** the most relevant parts of your document using semantic search
- **Generates** accurate, contextual answers using Google Gemini
- **Cites** exactly which page and document each answer came from
- **Remembers** your conversation history for follow-up questions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│   Dashboard | Upload | Chat | Search | About               │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP (httpx)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│   /upload  /chat  /search  /status  /documents             │
└──────────┬────────────────────────┬───────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐    ┌────────────────────────────────────┐
│  DOCUMENT        │    │         RAG PIPELINE               │
│  PROCESSOR       │    │                                    │
│                  │    │  Query → Embed → FAISS Search      │
│  PyPDF           │    │      → Retrieved Chunks            │
│  Text Splitter   │    │      → Gemini LLM                  │
│  → Chunks        │    │      → Answer + Citations          │
└──────────┬───────┘    └────────────────────────────────────┘
           │
           ▼
┌──────────────────┐
│  FAISS VECTOR    │
│  DATABASE        │
│                  │
│  sentence-       │
│  transformers    │
│  embeddings      │
│  (384-dim)       │
└──────────────────┘
```

### RAG Pipeline Explained

| Phase | What Happens |
|-------|-------------|
| **Indexing** | PDF → PyPDF → Chunks (1000 chars, 200 overlap) → Embeddings → FAISS |
| **Retrieval** | Question → Embed → FAISS ANN Search → Top-5 relevant chunks |
| **Generation** | Question + Chunks + History → Gemini prompt → Answer + Sources |

---

## ✨ Features

| Feature | Implementation |
|---------|---------------|
| 📄 Multi-PDF Upload | FastAPI file upload + PyPDF processing |
| 🧩 Intelligent Chunking | RecursiveCharacterTextSplitter with overlap |
| 🔢 Local Embeddings | sentence-transformers (all-MiniLM-L6-v2) — no API cost |
| 🗃️ Vector Database | FAISS with persistent disk storage |
| 🤖 LLM Generation | Google Gemini 1.5 Flash via LangChain |
| 💬 Chat Memory | ConversationBufferWindowMemory (last 5 exchanges) |
| 📎 Source Citations | Returns file + page for every answer |
| 🔍 Semantic Search | Pure embedding similarity search without LLM |
| 🔐 Authentication | Simple username/password (configurable via .env) |
| 📊 Dashboard | Real-time system status and stats |
| 📖 API Docs | Auto-generated Swagger UI at `/docs` |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Google Gemini 1.5 Flash | Answer generation |
| **RAG Framework** | LangChain 0.2 | Chain orchestration |
| **Vector DB** | FAISS (CPU) | Semantic similarity search |
| **Embeddings** | sentence-transformers | Local text embeddings (free) |
| **Backend** | FastAPI + Uvicorn | REST API server |
| **Frontend** | Streamlit | Interactive web UI |
| **PDF** | PyPDF + PyMuPDF | Text extraction |
| **Config** | pydantic-settings | Environment management |
| **HTTP Client** | httpx | Async API calls |

---

## 📁 Project Structure

```
enterprise-ai-assistant/
│
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — all REST endpoints
│   ├── config.py            # Centralized settings (pydantic-settings)
│   └── core/
│       ├── __init__.py
│       └── rag_pipeline.py  # Full RAG pipeline:
│                            #   DocumentProcessor
│                            #   VectorStoreManager (FAISS)
│                            #   LLMManager (Gemini)
│                            #   RAGChainBuilder (LangChain)
│                            #   RAGPipeline (orchestrator)
│
├── frontend/
│   ├── __init__.py
│   ├── app.py               # Streamlit UI (5 pages)
│   └── api_client.py        # HTTP client for FastAPI
│
├── data/                    # Auto-created at runtime
│   ├── uploads/             # Uploaded PDFs
│   ├── vectorstore/         # FAISS index files
│   └── processed/           # Processing metadata
│
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt
├── render.yaml              # Render.com deployment config
├── run_backend.sh           # Quick start script
├── run_frontend.sh          # Quick start script
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and Set Up

```bash
git clone https://github.com/yourusername/enterprise-ai-assistant.git
cd enterprise-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
AUTH_USERNAME=admin
AUTH_PASSWORD=your_password_here
```

> 🔑 Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Start the Backend

```bash
# Terminal 1
uvicorn backend.main:app --reload --port 8000
```

✅ API docs available at: **http://localhost:8000/docs**

### 4. Start the Frontend

```bash
# Terminal 2
streamlit run frontend/app.py
```

✅ App available at: **http://localhost:8501**

---

## 🌐 Deployment

### Deploy to Render.com (Free)

1. Push your code to GitHub
2. Sign up at [render.com](https://render.com)
3. New → Web Service → Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables in Render dashboard
7. Deploy! 🚀

### Deploy to HuggingFace Spaces

1. Create a new Space (Streamlit SDK)
2. Upload your files
3. Add `GEMINI_API_KEY` in Space secrets
4. The Streamlit app will run automatically

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/status` | System status + stats |
| `POST` | `/upload` | Upload PDF (multipart/form-data) |
| `POST` | `/chat` | Ask a question → answer + sources |
| `POST` | `/search` | Semantic search → ranked chunks |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents` | Clear all documents |
| `POST` | `/auth/login` | Authenticate → token |

Full interactive docs: **http://localhost:8000/docs**

---

## 💼 Resume Description

### Short Version (for resume bullet points):
> Built a **RAG-powered document Q&A system** using LangChain, FAISS, and Google Gemini API. Implemented semantic search with sentence-transformers embeddings, conversational memory, and source citations. Exposed via FastAPI REST API with Streamlit frontend.

### Full Version (for resume projects section):

**Enterprise AI Knowledge Assistant** | Python, LangChain, FAISS, FastAPI, Streamlit, Gemini API

- Designed and built an end-to-end **RAG (Retrieval-Augmented Generation) pipeline** enabling conversational Q&A over uploaded PDF documents
- Implemented **FAISS vector database** with sentence-transformers embeddings (all-MiniLM-L6-v2) for millisecond-level semantic similarity search across document chunks
- Built **LangChain ConversationalRetrievalChain** with window-based memory to maintain context across multi-turn conversations
- Developed **FastAPI REST backend** with 8 endpoints, dependency injection, Pydantic validation, and auto-generated Swagger documentation
- Created **Streamlit frontend** with 5 pages: dashboard, document upload, chat assistant, semantic search explorer, and architecture overview
- Implemented **source citation system** that traces every AI answer back to specific PDF pages
- Deployed on Render.com with persistent FAISS vector store

**Key Skills Demonstrated:** LLM APIs, RAG pipelines, vector databases, semantic search, embeddings, LangChain, FastAPI, Streamlit, REST APIs, Python

---


