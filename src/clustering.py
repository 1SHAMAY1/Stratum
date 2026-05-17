"""
Stratum RAPTOR Clustering Engine
==================================

Implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized
Retrieval) technique. Builds a hierarchical tree of knowledge layers from
a flat list of raw text chunks.
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from config.settings import settings

# Initialise the Gemini client once at module level.
_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _get_retry_delay(err: APIError) -> float | None:
    """Parse APIError message for specified retry delays (e.g. Please retry in 22.08s)."""
    if not err.message:
        return None
    match = re.search(r"Please retry in ([\d\.]+)s", err.message)
    if match:
        return float(match.group(1))
    return None


class RaptorTreeBuilder:
    """
    Builds a recursive RAPTOR knowledge tree from a list of text nodes.

    Attributes:
        max_clusters (int): Upper bound on K-Means clusters per layer.
        min_cluster_size (int): Minimum nodes needed to continue recursion.
        embedding_model (str): Gemini embedding model identifier.
        llm_model (str): Gemini LLM model identifier used for summarization.
    """

    def __init__(self) -> None:
        self.max_clusters: int = settings.MAX_CLUSTERS
        self.min_cluster_size: int = settings.MIN_CLUSTER_SIZE
        self.embedding_model: str = settings.EMBEDDING_MODEL
        self.llm_model: str = settings.SUMMARIZATION_MODEL

    # ── Public Methods ─────────────────────────────────────────────────────

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of text strings using batching.

        Uses smart exponential backoff & parses API-suggested delays for 429/503.
        """
        if not texts:
            return np.empty((0, 3072), dtype=np.float32)

        batch_size = 100
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            content_list = [
                types.Content(parts=[types.Part.from_text(text=txt)])
                for txt in batch_texts
            ]

            max_retries = 6
            base_delay = 2.0
            batch_embeddings = None

            for attempt in range(max_retries):
                try:
                    result = _client.models.embed_content(
                        model=self.embedding_model,
                        contents=content_list,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    batch_embeddings = [emb.values for emb in result.embeddings]
                    time.sleep(0.5)
                    break
                except APIError as e:
                    if e.code in (429, 503) and attempt < max_retries - 1:
                        delay = _get_retry_delay(e)
                        if delay is not None:
                            time.sleep(delay + 1.5)
                        else:
                            time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise e

            if batch_embeddings is None:
                raise RuntimeError("Failed to generate embeddings after retries.")
            embeddings.extend(batch_embeddings)

        return np.array(embeddings, dtype=np.float32)

    def build_tree_layer(
        self,
        nodes: list[dict[str, Any]],
        current_layer: int = 0,
    ) -> list[dict[str, Any]]:
        """Recursively build one layer of the RAPTOR knowledge tree in parallel."""
        if len(nodes) <= self.min_cluster_size:
            return []

        embeddings = np.array(
            [node["embedding"] for node in nodes], dtype=np.float32
        )
        k = self._determine_k(len(nodes))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels: np.ndarray = kmeans.fit_predict(embeddings)

        clusters: dict[int, list[dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(nodes[idx])

        summary_nodes: list[dict[str, Any]] = []
        next_layer: int = current_layer + 1

        def process_cluster(cluster_id: int, cluster_nodes: list[dict[str, Any]]) -> dict[str, Any]:
            combined_text = "\n\n".join(node["text"] for node in cluster_nodes)
            summary_text = self._generate_summary(combined_text, next_layer)
            summary_embedding = self.get_embeddings([summary_text])[0]
            return {
                "id": f"summary_L{next_layer}_C{cluster_id}",
                "text": summary_text,
                "embedding": summary_embedding.tolist(),
                "layer": next_layer,
                "is_summary": True,
                "source_node_ids": [n["id"] for n in cluster_nodes],
            }

        with ThreadPoolExecutor(max_workers=self.max_clusters) as executor:
            futures = [
                executor.submit(process_cluster, cid, cnodes)
                for cid, cnodes in clusters.items()
            ]
            for future in futures:
                summary_nodes.append(future.result())

        upper_summaries = self.build_tree_layer(summary_nodes, next_layer)
        return summary_nodes + upper_summaries

    # ── Private Methods ────────────────────────────────────────────────────

    def _determine_k(self, num_nodes: int) -> int:
        k = min(num_nodes // 2, self.max_clusters)
        return max(k, 2)

    def _generate_summary(self, combined_text: str, layer: int) -> str:
        """Generate high-level summary using Gemini LLM with retry delays on 429/503."""
        prompt = (
            f"You are summarizing a cluster of related technical content "
            f"at abstraction layer {layer} of a RAPTOR knowledge tree.\n\n"
            f"Write a comprehensive but concise summary that captures:\n"
            f"- Core concepts and their relationships\n"
            f"- Key architectural or design decisions\n"
            f"- Important technical details worth preserving\n\n"
            f"Source content:\n---\n{combined_text}\n---\n\n"
            f"Summary:"
        )
        max_retries = 6
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = _client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                )
                time.sleep(0.5)
                return response.text.strip()
            except APIError as e:
                if e.code in (429, 503) and attempt < max_retries - 1:
                    delay = _get_retry_delay(e)
                    if delay is not None:
                        time.sleep(delay + 1.5)
                    else:
                        time.sleep(base_delay * (2 ** attempt))
                else:
                    raise e
        return ""
