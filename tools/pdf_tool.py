"""
PDF Processor — extracts text and chunks from PDF files using PyMuPDF.
"""

import fitz
from tools.chunking_tool import TextChunker


class PDFProcessor:

    def __init__(self):
        self.chunker = TextChunker()

    def extract_text_and_chunks(self, file_path: str) -> dict:

        try:
            doc = fitz.open(file_path)

            all_chunks = []
            full_text = ""

            for page_number, page in enumerate(doc):

                page_text = page.get_text("text", sort=True)

                if not page_text or len(page_text.strip()) < 50:
                    continue

                full_text += page_text + "\n"

                chunks = self.chunker.chunk_text(page_text)

                for chunk in chunks:

                    if len(chunk.strip()) < 80:
                        continue

                    all_chunks.append({
                        "page_number": page_number + 1,
                        "text": chunk.strip()
                    })

            total_pages = len(doc)
            doc.close()

            return {
                "file_path": file_path,
                "text": full_text,
                "chunks": all_chunks,
                "total_pages": total_pages,
            }

        except FileNotFoundError:
            return {"error": f"PDF file not found: {file_path}"}

        except Exception as e:
            return {"error": f"PDF processing failed: {str(e)}"}