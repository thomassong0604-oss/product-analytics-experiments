"""Puts the repository root on the import path, so a bare `pytest` works after a fresh clone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
