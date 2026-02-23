import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib


class VectorStore:

    def __init__(self, collection_name="pdf_collection"):

        self.client = chromadb.PersistentClient(
                                                    path="vector_db"
                                                )


        self.collection = self.client.get_or_create_collection(collection_name)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _generate_id(self, text: str):
        return hashlib.md5(text.encode()).hexdigest()

    def add_documents(self, documents: list, metadata: list):

        if not documents:
            return

        embeddings = self.model.encode(documents).tolist()

        ids = [
                    f"{metadata[i].get('source','unknown')}_{metadata[i].get('chunk_id', i)}"
                    for i in range(len(documents))
                ]


        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )


    def query(self, query_text: str, top_k: int = 5):

        if not query_text:
            return {}

        embedding = self.model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k
        )

        return results
