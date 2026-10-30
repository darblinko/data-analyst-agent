#!/bin/bash
# data-analyst-agent - Launch Script (uses uv for environment management)

set -e

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your API keys, then run this script again."
    exit 1
fi

# Create uploads directory
mkdir -p uploads

echo ""
echo "Starting data-analyst-agent..."
echo "   URL: http://localhost:8501"
echo ""

# uv run automatically creates .venv and syncs deps from pyproject.toml
uv run streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false
