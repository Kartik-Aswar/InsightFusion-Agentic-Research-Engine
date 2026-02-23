import fitz
from tools.chunking_tool import TextChunker


class PDFProcessor:

    def __init__(self):
        self.chunker = TextChunker()

    def extract_text_and_chunks(self, file_path: str):

        try:
            doc = fitz.open(file_path)

            full_text = ""
            page_chunks = []

            for page_number, page in enumerate(doc):

                page_text = page.get_text()
                full_text += page_text

                chunks = self.chunker.chunk_text(page_text)

                for chunk in chunks:
                    page_chunks.append({
                        "page_number": page_number + 1,
                        "text": chunk
                    })

            doc.close()

            return {
                "file_path": file_path,
                "text": full_text,
                "chunks": page_chunks
            }

        except Exception as e:
            return {"error": str(e)}
