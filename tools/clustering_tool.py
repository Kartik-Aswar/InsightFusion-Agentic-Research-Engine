from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np


class InsightClusterer:
    """
    Clusters document chunks into thematic groups.
    """

    def __init__(self, model=None, max_clusters=5):
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.max_clusters = max_clusters

    def cluster(self, texts):

        if not texts:
            return {}

        if len(texts) == 1:
            return {0: texts}

        embeddings = self.model.encode(texts)

        n_clusters = min(self.max_clusters, len(texts))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clustered = {}
        for idx, label in enumerate(labels):
            clustered.setdefault(label, []).append(texts[idx])

        return clustered
