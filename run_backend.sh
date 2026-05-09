#!/bin/bash
# run_backend.sh — Start the FastAPI backend
# Usage: bash run_backend.sh

echo "🚀 Starting FastAPI Backend..."
echo "📖 API Docs available at: http://localhost:8000/docs"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start FastAPI with uvicorn
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir backend
