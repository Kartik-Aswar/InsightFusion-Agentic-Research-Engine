"""
Shared logging utility — Tee class for dual output (terminal + file).
Used by both main.py (CLI) and app.py (Streamlit).
"""

import sys
import os
from datetime import datetime


class Tee:
    """
    Duplicates writes to multiple file-like objects (e.g., stdout + log file).
    Compatible with Rich / CrewAI console rendering.
    """

    def __init__(self, *files):
        self.files = files
        self._original_stream = files[0]

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

    # Required for Rich / CrewAI console rendering
    def isatty(self):
        return self._original_stream.isatty()

    @property
    def encoding(self):
        return self._original_stream.encoding


def setup_logging(log_dir: str = "logs") -> str:
    """
    Initialize dual logging (terminal + file).
    Returns the log file path.
    """

    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime(f"{log_dir}/run_%Y%m%d_%H%M%S.log")
    log_file = open(log_filename, "w", encoding="utf-8")

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    return log_filename
