import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from backend.config import get_settings

settings = get_settings()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_total"] = len(chunks)
        return chunks

    def process_file(self, file_path: str) -> Tuple[List[Document], Dict]:
        documents = self.load_pdf(file_path)
        chunks = self.split_documents(documents)
        stats = {
            "file_name": Path(file_path).name,
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
        }
        return chunks, stats


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: Optional[FAISS] = None
        self.store_path = settings.vector_store_path
        self._load_existing_store()

    def _load_existing_store(self):
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
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        self.vector_store.save_local(self.store_path)
        return len(chunks)

    def similarity_search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        if self.vector_store is None:
            return []
        k = k or settings.top_k_results
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = None):
        if self.vector_store is None:
            return None
        k = k or settings.top_k_results
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def clear_store(self):
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


class LLMManager:
    def __init__(self):
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        if not settings.groq_api_key:
            print("⚠️  No GROQ_API_KEY found. LLM features disabled.")
            return
        self.llm = ChatGroq(
            model=settings.llm_model,
            groq_api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )
        print(f"✅ LLM initialized: {settings.llm_model}")

    @property
    def is_ready(self) -> bool:
        return self.llm is not None


SYSTEM_TEMPLATE = """You are an intelligent AI Knowledge Assistant. Answer questions based ONLY on the provided document context.

INSTRUCTIONS:
- Answer based strictly on the context provided below
- If the answer is not in the context, say "I couldn't find this information in the uploaded documents"
- Always be concise, clear, and helpful
- Cite which document/page your information comes from when possible

CONTEXT FROM DOCUMENTS:
{context}

CHAT HISTORY:
{chat_history}
"""

HUMAN_TEMPLATE = """Question: {question}

Please provide a detailed answer based on the document context above."""


class RAGChainBuilder:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chain = None
        self._build_chain()

    def _build_chain(self):
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=5,
        )
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=False,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        result = self.chain.invoke({"question": question})
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            source_key = f"{meta.get('source', 'Unknown')}_{meta.get('page', 0)}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "file": Path(meta.get("source", "Unknown")).name,
                    "page": meta.get("page", 0) + 1,
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                })
        return {
            "answer": result.get("answer", "I could not generate an answer."),
            "sources": sources,
        }


class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.rag_chain: Optional[RAGChainBuilder] = None
        self.uploaded_files: List[Dict] = []
        if self.vector_store_manager.is_ready and self.llm_manager.is_ready:
            self._rebuild_chain()

    def _rebuild_chain(self):
        retriever = self.vector_store_manager.get_retriever()
        if retriever and self.llm_manager.llm:
            self.rag_chain = RAGChainBuilder(self.llm_manager.llm, retriever)

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        chunks, stats = self.document_processor.process_file(file_path)
        self.vector_store_manager.add_documents(chunks)
        if self.llm_manager.is_ready:
            self._rebuild_chain()
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
        if not self.vector_store_manager.is_ready:
            return {"answer": "Please upload documents first.", "sources": [], "error": "no_documents"}
        if not self.llm_manager.is_ready:
            return {"answer": "API key not configured.", "sources": [], "error": "no_api_key"}
        if self.rag_chain is None:
            self._rebuild_chain()
        return self.rag_chain.ask(question)

    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
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
        self.vector_store_manager.clear_store()
        self.rag_chain = None
        self.uploaded_files = []

    def get_status(self) -> Dict[str, Any]:
        return {
            "vector_store_ready": self.vector_store_manager.is_ready,
            "llm_ready": self.llm_manager.is_ready,
            "rag_chain_ready": self.rag_chain is not None,
            "total_indexed_chunks": self.vector_store_manager.get_document_count(),
            "uploaded_files": self.uploaded_files,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }