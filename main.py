"""
data-analyst-agent - CLI Entry Point

Alternative entry point that launches Streamlit programmatically.
Primary usage: streamlit run app.py  or  ./run.sh
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the data-analyst-agent Streamlit app."""
    project_dir = Path(__file__).parent.resolve()
    app_path = project_dir / "app.py"

    print("Starting data-analyst-agent...")
    print("   URL: http://localhost:8501")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(project_dir),
    )


if __name__ == "__main__":
    main()
