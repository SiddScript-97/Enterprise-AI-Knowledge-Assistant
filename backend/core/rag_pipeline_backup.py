# backend/core/rag_pipeline.py
# ============================================================
# RAG (Retrieval-Augmented Generation) Pipeline
#
# This is the HEART of the project. Here's what it does:
#
# 1. INDEXING PHASE (when you upload docs):
#    PDF → Text Chunks → Embeddings → FAISS Vector Store
#
# 2. RETRIEVAL PHASE (when you ask a question):
#    Question → Embedding → Semantic Search → Top-K Chunks
#
# 3. GENERATION PHASE (creating the answer):
#    Question + Retrieved Chunks → LLM Prompt → Answer + Sources
#
# Interview Explanation:
# "RAG grounds the LLM in your documents. Instead of hallucinating,
#  it retrieves relevant context first, then generates an answer
#  based on that context. This makes responses factual and cited."
# ============================================================

import os
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from backend.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# 1. DOCUMENT PROCESSOR
# ─────────────────────────────────────────────

class DocumentProcessor:
    """
    Handles loading and chunking of PDF documents.

    Interview point: "I use RecursiveCharacterTextSplitter which
    splits on paragraphs first, then sentences, then words — this
    preserves semantic coherence within chunks better than naive
    character splitting."
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,       # ~1000 chars per chunk
            chunk_overlap=settings.chunk_overlap,  # 200 char overlap prevents context loss at boundaries
            separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF and return list of LangChain Document objects."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks with metadata preserved."""
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index metadata for better source tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_total"] = len(chunks)

        return chunks

    def process_file(self, file_path: str) -> Tuple[List[Document], Dict]:
        """Full pipeline: load → split → return chunks + stats."""
        documents = self.load_pdf(file_path)
        chunks = self.split_documents(documents)

        stats = {
            "file_name": Path(file_path).name,
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
        }

        return chunks, stats


# ─────────────────────────────────────────────
# 2. VECTOR STORE MANAGER
# ─────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages the FAISS vector database.

    Interview point: "FAISS (Facebook AI Similarity Search) stores
    document embeddings and enables millisecond-level similarity
    search using cosine similarity / L2 distance. I use
    sentence-transformers for embeddings because it's free,
    runs locally, and produces high-quality 384-dim vectors."
    """

    def __init__(self):
        # HuggingFace embeddings run locally — no API cost!
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )
        self.vector_store: Optional[FAISS] = None
        self.store_path = settings.vector_store_path
        self.metadata_path = os.path.join(self.store_path, "metadata.pkl")

        # Try loading existing store on init
        self._load_existing_store()

    def _load_existing_store(self):
        """Load FAISS index from disk if it exists."""
        index_path = os.path.join(self.store_path, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ Loaded existing vector store from {self.store_path}")
            except Exception as e:
                print(f"⚠️  Could not load vector store: {e}")

    def add_documents(self, chunks: List[Document]) -> int:
        """
        Add document chunks to FAISS.
        Creates new store if none exists, otherwise merges.
        """
        if self.vector_store is None:
            # First-time: create fresh FAISS index
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            # Subsequent uploads: add to existing index
            self.vector_store.add_documents(chunks)

        # Persist to disk so it survives server restarts
        self.vector_store.save_local(self.store_path)
        return len(chunks)

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Semantic search: find top-k most similar chunks to query.

        Interview point: "This converts the query to an embedding
        vector, then does approximate nearest-neighbor search in
        the FAISS index — much faster than comparing every document."
        """
        if self.vector_store is None:
            return []
        k = k or settings.top_k_results
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Returns (document, relevance_score) pairs for transparency."""
        if self.vector_store is None:
            return []
        k = k or settings.top_k_results
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = None):
        """Return LangChain-compatible retriever for use in chains."""
        if self.vector_store is None:
            return None
        k = k or settings.top_k_results
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def clear_store(self):
        """Delete all indexed documents."""
        self.vector_store = None
        import shutil
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
            os.makedirs(self.store_path, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None

    def get_document_count(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal


# ─────────────────────────────────────────────
# 3. LLM MANAGER
# ─────────────────────────────────────────────

class LLMManager:
    """
    Manages the Language Model (Google Gemini).

    Interview point: "I use Gemini-1.5-flash which is fast and
    cost-effective. The temperature=0.3 keeps answers factual
    rather than creative — perfect for a Q&A assistant."
    """

    def __init__(self):
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize Gemini LLM via LangChain integration."""
        if not settings.gemini_api_key:
            print("⚠️  No GEMINI_API_KEY found. LLM features disabled.")
            return

        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.max_tokens,
            convert_system_message_to_human=True,  # Gemini compatibility
        )
        print(f"✅ LLM initialized: {settings.llm_model}")

    @property
    def is_ready(self) -> bool:
        return self.llm is not None


# ─────────────────────────────────────────────
# 4. RAG CHAIN BUILDER
# ─────────────────────────────────────────────

# Custom prompt template — this is what gets sent to Gemini
SYSTEM_TEMPLATE = """You are an intelligent AI Knowledge Assistant. Your job is to answer questions \
based ONLY on the provided document context. 

INSTRUCTIONS:
- Answer based strictly on the context provided below
- If the answer is not in the context, say "I couldn't find this information in the uploaded documents"
- Always be concise, clear, and helpful
- Cite which document/page your information comes from when possible
- If asked for a summary, provide a structured response

CONTEXT FROM DOCUMENTS:
{context}

CHAT HISTORY:
{chat_history}
"""

HUMAN_TEMPLATE = """Question: {question}

Please provide a detailed answer based on the document context above."""


class RAGChainBuilder:
    """
    Builds the complete Conversational RAG chain.

    Interview point: "ConversationalRetrievalChain combines three things:
    1. A retriever that fetches relevant chunks for the question
    2. Conversation memory that tracks chat history
    3. An LLM that generates the final answer
    This gives us context-aware, memory-enabled Q&A over documents."
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chain = None
        self._build_chain()

    def _build_chain(self):
        """Assemble the LangChain ConversationalRetrievalChain."""
        # ConversationBufferWindowMemory keeps last k exchanges
        # This prevents context window overflow on long chats
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=5,  # remember last 5 Q&A pairs
        )

        # Build the chat prompt
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        # ConversationalRetrievalChain = retriever + memory + LLM
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,  # crucial for citations
            verbose=False,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer + source citations.

        Returns:
            {
                "answer": "The answer text...",
                "sources": [{"file": "doc.pdf", "page": 2, "content": "..."}]
            }
        """
        result = self.chain.invoke({"question": question})

        # Extract source citations from retrieved documents
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            source_key = f"{meta.get('source', 'Unknown')}_{meta.get('page', 0)}"

            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "file": Path(meta.get("source", "Unknown")).name,
                    "page": meta.get("page", 0) + 1,  # 0-indexed → 1-indexed
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                })

        return {
            "answer": result.get("answer", "I could not generate an answer."),
            "sources": sources,
        }


# ─────────────────────────────────────────────
# 5. MAIN RAG PIPELINE (orchestrator)
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Top-level orchestrator that ties everything together.
    This is the single class imported by the FastAPI app.

    Interview point: "I designed this as a singleton-like class
    that the FastAPI app instantiates once. It holds the vector
    store in memory for fast retrieval, rebuilding the chain
    whenever new documents are added."
    """

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.rag_chain: Optional[RAGChainBuilder] = None
        self.uploaded_files: List[Dict] = []

        # If vector store already has data, build chain immediately
        if self.vector_store_manager.is_ready and self.llm_manager.is_ready:
            self._rebuild_chain()

    def _rebuild_chain(self):
        """(Re)build the RAG chain after adding documents."""
        retriever = self.vector_store_manager.get_retriever()
        if retriever and self.llm_manager.llm:
            self.rag_chain = RAGChainBuilder(self.llm_manager.llm, retriever)

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Full document ingestion pipeline:
        PDF file → chunks → embeddings → FAISS index

        Called by the FastAPI /upload endpoint.
        """
        start_time = time.time()

        # Step 1: Load and chunk the PDF
        chunks, stats = self.document_processor.process_file(file_path)

        # Step 2: Store embeddings in FAISS
        self.vector_store_manager.add_documents(chunks)

        # Step 3: Rebuild chain with updated vector store
        if self.llm_manager.is_ready:
            self._rebuild_chain()

        # Track uploaded files
        self.uploaded_files.append({
            "name": stats["file_name"],
            "pages": stats["total_pages"],
            "chunks": stats["total_chunks"],
            "path": file_path,
        })

        elapsed = round(time.time() - start_time, 2)
        return {
            "success": True,
            "file_name": stats["file_name"],
            "pages_processed": stats["total_pages"],
            "chunks_created": stats["total_chunks"],
            "avg_chunk_size": stats["avg_chunk_size"],
            "processing_time_sec": elapsed,
            "total_indexed_chunks": self.vector_store_manager.get_document_count(),
        }

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        Called by FastAPI /chat endpoint.
        """
        if not self.vector_store_manager.is_ready:
            return {
                "answer": "Please upload documents first before asking questions.",
                "sources": [],
                "error": "no_documents",
            }

        if not self.llm_manager.is_ready:
            return {
                "answer": "API key not configured. Please set GEMINI_API_KEY in your .env file.",
                "sources": [],
                "error": "no_api_key",
            }

        if self.rag_chain is None:
            self._rebuild_chain()

        return self.rag_chain.ask(question)

    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Pure semantic search without LLM generation.
        Returns raw matching chunks with relevance scores.
        """
        results = self.vector_store_manager.similarity_search_with_scores(query, k=k)
        return [
            {
                "content": doc.page_content,
                "file": Path(doc.metadata.get("source", "Unknown")).name,
                "page": doc.metadata.get("page", 0) + 1,
                "score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def clear_all_documents(self):
        """Reset the entire knowledge base."""
        self.vector_store_manager.clear_store()
        self.rag_chain = None
        self.uploaded_files = []

    def get_status(self) -> Dict[str, Any]:
        """Health check / status for the frontend dashboard."""
        return {
            "vector_store_ready": self.vector_store_manager.is_ready,
            "llm_ready": self.llm_manager.is_ready,
            "rag_chain_ready": self.rag_chain is not None,
            "total_indexed_chunks": self.vector_store_manager.get_document_count(),
            "uploaded_files": self.uploaded_files,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }
