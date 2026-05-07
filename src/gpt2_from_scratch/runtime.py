"""Small runtime helpers for cross-platform scripts."""

import os
import sys


def configure_runtime() -> None:
    """Apply local runtime defaults before importing heavy ML libraries."""
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

