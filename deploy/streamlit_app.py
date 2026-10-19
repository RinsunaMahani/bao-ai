"""Entry point for Streamlit Community Cloud.

Community Cloud installs the requirements file beside the entry point, so
this folder holds one that adds the language classifier and PDF reading
to the core requirements. The app itself is bao/ui/streamlit_app.py; this
only runs it, from the repository root, as `streamlit run` would.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "bao" / "ui" / "streamlit_app.py"), run_name="__main__")
