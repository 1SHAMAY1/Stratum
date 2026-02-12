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
        self.llm_model = settings.LLM_MODEL

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

    def _generate_summary(self, combined_text: str, layer: int) -> str:
        prompt = f"Summarize layer {layer}:\n{combined_text}"
        response = _client.models.generate_content(model=self.llm_model, contents=prompt)
        return response.text.strip()

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
        summary_nodes = []
        next_layer = current_layer + 1
        for cluster_id, cluster_nodes in clusters.items():
            combined_text = "\n\n".join(node["text"] for node in cluster_nodes)
            summary_text = self._generate_summary(combined_text, next_layer)
            summary_embedding = self.get_embeddings([summary_text])[0]
            summary_node = {
                "id": f"summary_L{next_layer}_C{cluster_id}",
                "text": summary_text,
                "embedding": summary_embedding.tolist(),
                "layer": next_layer,
                "is_summary": True,
                "source_node_ids": [n["id"] for n in cluster_nodes],
            }
            summary_nodes.append(summary_node)
        upper_summaries = self.build_tree_layer(summary_nodes, next_layer)
        return summary_nodes + upper_summaries
