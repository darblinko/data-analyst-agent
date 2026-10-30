"""Streamlit entry point: `uv run streamlit run app.py` or `./run.sh`."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from data_analyst_agent.ui.streamlit_app import main  # noqa: E402

main()
