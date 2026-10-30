#!/usr/bin/env bash
# Start the Streamlit app on http://localhost:8501. Extra arguments go to streamlit run.
# DEMO_MODE=1 ./run.sh starts the scripted demo, which needs no API key.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    "$@"
