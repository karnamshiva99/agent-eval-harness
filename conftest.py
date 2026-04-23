"""Ensures `src/` is importable during pytest collection without requiring
the package to be installed. Keeps the quickstart to a single pip install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
