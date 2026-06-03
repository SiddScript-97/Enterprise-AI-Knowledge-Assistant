#!/bin/bash
# run_frontend.sh — Start the Streamlit frontend
# Usage: bash run_frontend.sh

echo "🎨 Starting Streamlit Frontend..."
echo "🌐 Open your browser at: http://localhost:8501"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

streamlit run frontend/app.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base dark
