# frontend/api_client.py
# ============================================================
# API Client — Handles all HTTP calls from Streamlit to FastAPI
#
# Interview point: "I separated the API client from the UI code
# to follow the Single Responsibility Principle. The frontend
# only does UI; the client handles HTTP communication."
# ============================================================

import httpx
import streamlit as st
from typing import Optional, Dict, Any, List

BACKEND_URL = "http://localhost:8000"
TIMEOUT = 60.0  # seconds (PDF processing can take time)


class APIClient:
    """
    Thin HTTP client wrapping the FastAPI backend.
    All methods return (data, error) tuples for clean error handling.
    """

    def __init__(self, base_url: str = BACKEND_URL, token: Optional[str] = None):
        self.base_url = base_url
        self.token = token

    @property
    def headers(self) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def login(self, username: str, password: str) -> tuple:
        """Authenticate and get a token."""
        try:
            response = httpx.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("token")
                return data, None
            return None, response.json().get("detail", "Login failed")
        except Exception as e:
            return None, f"Connection error: {str(e)}"

    def get_status(self) -> tuple:
        """Get system status."""
        try:
            response = httpx.get(f"{self.base_url}/status", timeout=10.0)
            if response.status_code == 200:
                return response.json(), None
            return None, "Failed to get status"
        except Exception as e:
            return None, f"Backend offline: {str(e)}"

    def upload_document(self, file_bytes: bytes, filename: str) -> tuple:
        """Upload a PDF file to the backend."""
        try:
            response = httpx.post(
                f"{self.base_url}/upload",
                files={"file": (filename, file_bytes, "application/pdf")},
                timeout=TIMEOUT,
            )
            if response.status_code == 200:
                return response.json(), None
            return None, response.json().get("detail", "Upload failed")
        except Exception as e:
            return None, f"Upload error: {str(e)}"

    def chat(self, question: str, session_id: str = "default") -> tuple:
        """Send a question and get an AI-generated answer."""
        try:
            response = httpx.post(
                f"{self.base_url}/chat",
                json={"question": question, "session_id": session_id},
                timeout=TIMEOUT,
            )
            if response.status_code == 200:
                return response.json(), None
            return None, response.json().get("detail", "Chat error")
        except Exception as e:
            return None, f"Chat error: {str(e)}"

    def semantic_search(self, query: str, top_k: int = 5) -> tuple:
        """Perform semantic search without LLM generation."""
        try:
            response = httpx.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json(), None
            return None, response.json().get("detail", "Search error")
        except Exception as e:
            return None, f"Search error: {str(e)}"

    def get_documents(self) -> tuple:
        """Get list of uploaded documents."""
        try:
            response = httpx.get(f"{self.base_url}/documents", timeout=10.0)
            if response.status_code == 200:
                return response.json(), None
            return None, "Failed to get documents"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def clear_documents(self) -> tuple:
        """Clear all documents from the knowledge base."""
        try:
            response = httpx.delete(f"{self.base_url}/documents", timeout=15.0)
            if response.status_code == 200:
                return response.json(), None
            return None, "Failed to clear documents"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def health_check(self) -> bool:
        """Quick check if backend is reachable."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False


# Singleton client stored in Streamlit session state
def get_client() -> APIClient:
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client
