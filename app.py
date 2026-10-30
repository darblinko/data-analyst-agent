"""
data-analyst-agent - Entry Point

Run with:
    streamlit run app.py

Or use the launch script:
    ./run.sh
"""

import sys
import os
from pathlib import Path

# Ensure the project directory is on the Python path so `src.*` imports work
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables before importing anything else
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Run the Streamlit app
from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()
