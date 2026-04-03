"""
Vector Store — ChromaDB + SentenceTransformer for PDF chunk storage and retrieval.

Uses cosine similarity (ChromaDB default) for semantic search.
"""

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self, collection_name="pdf_collection"):

        self.client = chromadb.PersistentClient(path="vector_db")

        self.collection = self.client.get_or_create_collection(collection_name)

        # Embedding model (384-dim, fast, good for academic text)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------------------------------
    # ADD DOCUMENTS
    # -------------------------------------------------

    def add_documents(self, documents: list, metadata: list) -> None:

        if not documents:
            return

        embeddings = self.model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        ids = [
            f"{metadata[i].get('source','unknown')}_{metadata[i].get('chunk_id', i)}"
            for i in range(len(documents))
        ]

        # UPSERT prevents duplicate ID errors on re-runs
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

    # -------------------------------------------------
    # QUERY VECTOR DATABASE
    # -------------------------------------------------

    def query(self, query_text: str, top_k: int = 25) -> dict:
        """
        Semantic search using cosine similarity (ChromaDB default).
        Returns top_k most relevant chunks.
        """

        if not query_text:
            return {}

        query_embedding = self.model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        # Safety check
        if not results or "documents" not in results:
            return {"documents": [[]]}

        if results["documents"] is None:
            return {"documents": [[]]}

        return results