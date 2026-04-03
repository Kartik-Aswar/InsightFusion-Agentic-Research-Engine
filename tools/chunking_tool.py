"""
Robust chunker for academic PDFs.
Produces clean semantic chunks for embedding.

FIX: Paragraph splitting now happens BEFORE whitespace normalization,
so double-newline boundaries are preserved correctly.
"""

from typing import List
import re


class TextChunker:

    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:

        if not text:
            return []

        # STEP 1: Split by paragraph boundaries FIRST (before normalizing)
        paragraphs = re.split(r"\n{2,}", text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:

            # STEP 2: Normalize whitespace WITHIN each paragraph
            para = re.sub(r"\s+", " ", para).strip()

            if not para:
                continue

            # If paragraph fits → append
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += " " + para
                continue

            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap for context continuity
            overlap_text = current_chunk[-self.overlap:] if current_chunk else ""
            current_chunk = overlap_text + " " + para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter tiny chunks that won't be useful for retrieval
        return [c.strip() for c in chunks if len(c) > 100]