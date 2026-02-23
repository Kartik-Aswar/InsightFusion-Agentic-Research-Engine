from typing import List
import re


class TextChunker:
    """
    Splits large text into semantically meaningful chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:

        if not text:
            return []

        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:

            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())

                # Overlap mechanism
                overlap_text = current_chunk[-self.overlap:]
                current_chunk = overlap_text + " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 50]
