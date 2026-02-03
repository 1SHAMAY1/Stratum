import numpy as np
from typing import Any
from sklearn.cluster import KMeans
from google import genai
from google.genai import types
from config.settings import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)

class RaptorTreeBuilder:
    def __init__(self) -> None:
        self.max_clusters = settings.MAX_CLUSTERS
        self.min_cluster_size = settings.MIN_CLUSTER_SIZE
        self.embedding_model = settings.EMBEDDING_MODEL

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            result = _client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.append(result.embeddings[0].values)
        return np.array(embeddings, dtype=np.float32)

    def _determine_k(self, num_nodes: int) -> int:
        k = min(num_nodes // 2, self.max_clusters)
        return max(k, 2)

    def build_tree_layer(self, nodes: list[dict[str, Any]], current_layer: int = 0) -> list[dict[str, Any]]:
        if len(nodes) <= self.min_cluster_size:
            return []
        embeddings = np.array([node["embedding"] for node in nodes], dtype=np.float32)
        k = self._determine_k(len(nodes))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(nodes[idx])
        return []
