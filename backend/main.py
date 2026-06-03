# backend/main.py
# ============================================================
# FastAPI Application - Main Entry Point
#
# REST API endpoints:
#   POST /upload        → Upload and index a PDF
#   POST /chat          → Ask a question (RAG Q&A)
#   POST /search        → Semantic search (no LLM)
#   GET  /status        → System health check
#   GET  /documents     → List uploaded documents
#   DELETE /documents   → Clear all documents
#   POST /auth/login    → Simple login (optional)
#
# Interview point: "I used FastAPI because it's production-grade,
# auto-generates OpenAPI docs at /docs, and is async-native.
# The RAG pipeline is instantiated once at startup and shared
# across all requests — this is the Singleton pattern."
# ============================================================

import os
import shutil
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import get_settings, ensure_directories
from backend.core.rag_pipeline import RAGPipeline

settings = get_settings()

# ─────────────────────────────────────────────
# Application State (shared across requests)
# ─────────────────────────────────────────────

# Global RAG pipeline instance — initialized once at startup
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    'lifespan' replaces the old @app.on_event("startup") pattern.
    """
    global rag_pipeline

    # --- STARTUP ---
    print("🚀 Starting Enterprise AI Knowledge Assistant...")
    ensure_directories()
    rag_pipeline = RAGPipeline()
    print("✅ RAG Pipeline initialized")
    print(f"📊 Status: {rag_pipeline.get_status()}")

    yield  # Server runs here

    # --- SHUTDOWN ---
    print("🛑 Shutting down...")


# ─────────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Enterprise AI Knowledge Assistant API",
    description="RAG-powered document Q&A system with semantic search",
    version="1.0.0",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc at /redoc
    lifespan=lifespan,
)

# CORS — allows the Streamlit frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic Models (Request/Response schemas)
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

    class Config:
        json_schema_extra = {
            "example": {"question": "What is the main topic of the document?"}
        }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

    class Config:
        json_schema_extra = {
            "example": {"query": "machine learning algorithms", "top_k": 5}
        }


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    processing_time: float


class SearchResponse(BaseModel):
    results: list
    query: str
    total_results: int


# ─────────────────────────────────────────────
# Dependency: Get RAG Pipeline
# ─────────────────────────────────────────────

def get_rag_pipeline() -> RAGPipeline:
    """FastAPI dependency injection for the RAG pipeline."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Enterprise AI Knowledge Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check for deployment monitoring."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/status", tags=["System"])
async def get_status(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Get full system status including vector store and LLM readiness.
    Used by the Streamlit frontend to show the dashboard metrics.
    """
    return pipeline.get_status()


# ── DOCUMENT UPLOAD ────────────────────────────────────────

@app.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Upload a PDF and add it to the knowledge base.

    Process:
    1. Save PDF to disk
    2. Extract text with PyPDF
    3. Split into chunks
    4. Generate embeddings with sentence-transformers
    5. Store in FAISS index

    Returns processing statistics.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Check file size (limit to 50MB)
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    # Save to upload directory
    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # Run ingestion pipeline
    try:
        result = pipeline.ingest_document(file_path)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# ── CONVERSATIONAL Q&A ────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Ask a question and get an AI-generated answer with source citations.

    This endpoint runs the full RAG pipeline:
    1. Retrieve relevant document chunks via semantic search
    2. Pass question + chunks to Gemini LLM
    3. Return answer with source citations

    The chat has memory — previous Q&A pairs are included in context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    try:
        result = pipeline.query(request.question.strip())
        elapsed = round(time.time() - start_time, 3)

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            processing_time=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ── SEMANTIC SEARCH ────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(
    request: SearchRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Pure semantic search — returns relevant document chunks WITHOUT LLM generation.

    Useful for exploring document content or debugging retrieval quality.
    Returns similarity scores for transparency.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = pipeline.semantic_search(
        query=request.query.strip(),
        k=min(request.top_k, 10),  # cap at 10
    )

    return SearchResponse(
        results=results,
        query=request.query,
        total_results=len(results),
    )


# ── DOCUMENT MANAGEMENT ───────────────────────────────────

@app.get("/documents", tags=["Documents"])
async def list_documents(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """List all uploaded and indexed documents."""
    status_data = pipeline.get_status()
    return {
        "documents": status_data["uploaded_files"],
        "total_chunks": status_data["total_indexed_chunks"],
    }


@app.delete("/documents", tags=["Documents"])
async def clear_documents(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Clear all uploaded documents and reset the vector store.
    Also cleans up uploaded files from disk.
    """
    pipeline.clear_all_documents()

    # Clean up uploaded files
    upload_dir = "./data/uploads"
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        os.makedirs(upload_dir)

    return {"message": "All documents cleared successfully", "success": True}


# ── AUTHENTICATION ─────────────────────────────────────────

@app.post("/auth/login", tags=["Auth"])
async def login(request: LoginRequest):
    """
    Simple username/password authentication.
    Returns a token (basic implementation for demo purposes).

    Interview point: "In production I'd use JWT tokens with
    proper expiry and refresh logic. For this project I kept
    it simple to focus on the AI features."
    """
    if (
        request.username == settings.auth_username
        and request.password == settings.auth_password
    ):
        # In production: generate a proper JWT token here
        import hashlib
        token = hashlib.sha256(
            f"{request.username}{settings.secret_key}{time.time()}".encode()
        ).hexdigest()

        return {
            "success": True,
            "token": token,
            "username": request.username,
            "message": "Login successful",
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


# ─────────────────────────────────────────────
# Run directly (for local development)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,  # auto-reload on code changes during dev
    )
