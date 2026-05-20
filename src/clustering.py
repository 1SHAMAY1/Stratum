"""
Stratum RAPTOR Clustering Engine
==================================

Implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized
Retrieval) technique. Builds a hierarchical tree of knowledge layers from
a flat list of raw text chunks. Emits exhaustive diagnostic logs to stdout.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
import warnings
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from config.settings import settings

# ──────────────────────────────────────────────────────────────────────────────
# Safe Console Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("stratum.clustering")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def _get_retry_delay(err: APIError) -> float | None:
    """Parse APIError message for specified retry delays (e.g. Please retry in 22.08s).

    Args:
        err (APIError): Raw Google API Exception.

    Returns:
        float | None: Number of seconds recommended by server, or None if unspecified.
    """
    if not err.message:
        return None
    match = re.search(r"Please retry in ([\d\.]+)s", err.message)
    if match:
        return float(match.group(1))
    return None


class APIRateLimiter:
    """Sliding-window thread-safe rate limiter to prevent 429 Quota limits on Free Tier keys.

    Attributes:
        max_limit (int): Maximum token actions allowed within the sliding window.
        window_seconds (float): Length of sliding window in seconds.
        history (list[float]): Monotonically increasing request timestamps.
    """

    def __init__(self, max_limit: int, window_seconds: float = 60.0) -> None:
        """Initialize the rate limiter configuration.

        Args:
            max_limit (int): Quota boundary threshold.
            window_seconds (float): Evaluation sliding window length.
        """
        self.max_limit: int = max_limit
        self.window_seconds: float = window_seconds
        self.history: list[float] = []
        self._lock: threading.Lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> None:
        """Acquire the specified tokens, pausing execution if we risk rate-limiting.

        Args:
            tokens (int): Amount of API requests to check.
        """
        if tokens <= 0:
            return
        if tokens > self.max_limit:
            # Clip if a single batch exceeds the total absolute capacity
            tokens = self.max_limit

        while True:
            with self._lock:
                now = time.time()
                # Evict stale timestamps
                self.history = [t for t in self.history if t > now - self.window_seconds]

                # If we have space, append immediately and continue
                if len(self.history) + tokens <= self.max_limit:
                    for _ in range(tokens):
                        self.history.append(now)
                    return

                # Calculate the wait time until the oldest requests slide out of the window
                needed_index = len(self.history) + tokens - self.max_limit - 1
                oldest_time = self.history[needed_index] if 0 <= needed_index < len(self.history) else now

            wait_time = oldest_time + self.window_seconds - now
            if wait_time > 0:
                logger.info(
                    "Rate limit safety trigger: Pausing for %.2f seconds to prevent 429 quota overflow...",
                    wait_time
                )
                time.sleep(wait_time)
            else:
                time.sleep(0.5)


# Shared API rate limiters to govern free tier quotas across the entire application
embedding_limiter = APIRateLimiter(max_limit=80, window_seconds=60.0)
llm_limiter = APIRateLimiter(max_limit=12, window_seconds=60.0)


class RaptorTreeBuilder:
    """
    Builds a recursive RAPTOR knowledge tree from a list of text nodes.

    Attributes:
        max_clusters (int): Upper bound on K-Means clusters per layer.
        min_cluster_size (int): Minimum nodes needed to continue recursion.
        embedding_model (str): Gemini embedding model identifier.
        llm_model (str): Gemini LLM model identifier used for summarization.
        client (genai.Client): The instantiated Gemini Client.
    """

    def __init__(self) -> None:
        """Initialize parameters and prepare models with lazy client construction."""
        self.max_clusters: int = settings.MAX_CLUSTERS
        self.min_cluster_size: int = settings.MIN_CLUSTER_SIZE
        self.embedding_model: str = settings.EMBEDDING_MODEL
        self.llm_model: str = settings.SUMMARIZATION_MODEL
        
        # Instantiate the client lazily on object initialization (NOT at import time)
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        logger.info(
            "RaptorTreeBuilder initialized | Models: [Embed: %s, LLM: %s] | Max Clusters: %d | Stop Size: %d",
            self.embedding_model,
            self.llm_model,
            self.max_clusters,
            self.min_cluster_size
        )

    # ── Public Methods ─────────────────────────────────────────────────────

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of text strings using batching.

        Uses an optimized batch layout, standard console logging, and adaptive backoff for rate limits.

        Args:
            texts (list[str]): Raw string elements to embed.

        Returns:
            np.ndarray: Calculated 2D float array containing vectors.
        """
        if not texts:
            logger.warning("Empty list of texts passed to get_embeddings. Returning blank array.")
            return np.empty((0, 3072), dtype=np.float32)

        # Batch size of 20 allows smooth pacing below the 100 RPM quota
        batch_size = 20
        embeddings: list[list[float]] = []
        total_items = len(texts)

        logger.info("Generating embeddings for %d text chunks in batches of %d", total_items, batch_size)

        for idx, i in enumerate(range(0, total_items, batch_size)):
            batch_texts = texts[i : i + batch_size]
            num_tokens = len(batch_texts)

            # Wait if acquiring embeddings exceeds our rate limit
            embedding_limiter.acquire(num_tokens)

            logger.info(
                "Embedding batch %d/%d (Chunks %d-%d)...",
                idx + 1,
                (total_items + batch_size - 1) // batch_size,
                i,
                i + num_tokens,
            )
            
            content_list = [
                types.Content(parts=[types.Part.from_text(text=txt)])
                for txt in batch_texts
            ]

            max_retries = 6
            base_delay = 2.0
            batch_embeddings = None

            for attempt in range(max_retries):
                try:
                    t_start = time.perf_counter()
                    result = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=content_list,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    duration = time.perf_counter() - t_start
                    batch_embeddings = [emb.values for emb in result.embeddings]
                    logger.debug("Successfully generated batch %d embeddings in %.2fs", idx + 1, duration)
                    break
                except APIError as e:
                    logger.warning(
                        "[API Warning] Embeddings batch %d failed on attempt %d/%d (Code %d). Message: %s",
                        idx + 1, attempt + 1, max_retries, e.code, e.message
                    )
                    if e.code in (429, 503) and attempt < max_retries - 1:
                        delay = _get_retry_delay(e)
                        sleep_time = delay + 1.5 if delay is not None else base_delay * (2 ** attempt)
                        logger.info("Applying exponential backoff: Sleeping for %.2f seconds...", sleep_time)
                        time.sleep(sleep_time)
                    else:
                        logger.error("[API Error] Unrecoverable exception generated by Gemini API.")
                        raise e

            if batch_embeddings is None:
                raise RuntimeError("Failed to generate embeddings after retries.")
            embeddings.extend(batch_embeddings)

        out_array = np.array(embeddings, dtype=np.float32)
        logger.info("Successfully vectorized entire batch. Dimensions generated: %s", out_array.shape)
        return out_array

    def build_tree_layer(
        self,
        nodes: list[dict[str, Any]],
        current_layer: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Recursively build one layer of the RAPTOR knowledge tree using linear orchestration.

        Args:
            nodes (list[dict[str, Any]]): Base list containing nodes with text and vectors.
            current_layer (int): Abstraction layer height index.

        Returns:
            list[dict[str, Any]]: Calculated summary nodes for current and upper layers.
        """
        logger.info("--- Building RAPTOR Layer %d ---", current_layer + 1)
        logger.info("Received %d nodes for layer construction.", len(nodes))

        if len(nodes) <= self.min_cluster_size:
            logger.info(
                "Active node length (%d) is <= Minimum cluster size (%d). Halting layer recursion.",
                len(nodes), self.min_cluster_size
            )
            return []

        embeddings = np.array(
            [node["embedding"] for node in nodes], dtype=np.float32
        )
        k = self._determine_k(len(nodes))
        logger.info("Fitting K-Means to divide nodes into k=%d clusters", k)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels: np.ndarray = kmeans.fit_predict(embeddings)

        clusters: dict[int, list[dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(nodes[idx])

        logger.info("K-Means separation completed. Discovered %d clusters.", len(clusters))

        summary_results: list[tuple[int, str, list[dict[str, Any]]]] = []
        next_layer: int = current_layer + 1

        # Step 1: Sequential summaries guided by the llm_limiter
        for cid, cnodes in clusters.items():
            combined_text = "\n\n".join(node["text"] for node in cnodes)
            logger.info(
                "Summarizing cluster %d/%d containing %d child nodes...",
                cid + 1, len(clusters), len(cnodes)
            )
            
            # Rate limit before calling LLM
            llm_limiter.acquire(1)
            
            t_start = time.perf_counter()
            summary_text = self._generate_summary(combined_text, next_layer)
            duration = time.perf_counter() - t_start
            logger.info("Cluster %d/%d summarized successfully in %.2fs", cid + 1, len(clusters), duration)
            
            summary_results.append((cid, summary_text, cnodes))

        # Step 2: Extract text summaries and perform unified batch embedding (guided inside get_embeddings by embedding_limiter)
        logger.info("Initiating embedding pass for all summaries generated in Layer %d...", next_layer)
        all_summary_texts = [res[1] for res in summary_results]
        all_summary_embeddings = self.get_embeddings(all_summary_texts)

        # Step 3: Package into final tree node formatting
        summary_nodes: list[dict[str, Any]] = []
        for (cid, summary_text, cnodes), summary_emb in zip(summary_results, all_summary_embeddings):
            summary_nodes.append({
                "id": f"summary_L{next_layer}_C{cid}",
                "text": summary_text,
                "embedding": summary_emb.tolist(),
                "layer": next_layer,
                "is_summary": True,
                "source_node_ids": [n["id"] for n in cnodes],
            })

        logger.info("Layer %d successfully assembled. Built %d summary nodes.", next_layer, len(summary_nodes))

        # Recurse upwards to continue organizing
        upper_summaries = self.build_tree_layer(summary_nodes, next_layer)
        return summary_nodes + upper_summaries

    # ── Private Methods ────────────────────────────────────────────────────

    def _determine_k(self, num_nodes: int) -> int:
        """Calculate optimal cluster dimension based on node count.

        Args:
            num_nodes (int): Total elements to divide.

        Returns:
            int: Number of clusters (k).
        """
        k = min(num_nodes // 2, self.max_clusters)
        return max(k, 2)

    def _generate_summary(self, combined_text: str, layer: int) -> str:
        """Generate high-level summary using Gemini LLM with retry delays on 429/503.

        Args:
            combined_text (str): Input text grouped together.
            layer (int): Abstraction layer level.

        Returns:
            str: Generated summary string.
        """
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
                response = self.client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                )
                return response.text.strip()
            except APIError as e:
                logger.warning(
                    "[API Warning] Summary generation failed on attempt %d/%d (Code %d). Message: %s",
                    attempt + 1, max_retries, e.code, e.message
                )
                if e.code in (429, 503) and attempt < max_retries - 1:
                    delay = _get_retry_delay(e)
                    sleep_time = delay + 1.5 if delay is not None else base_delay * (2 ** attempt)
                    logger.info("Applying backoff delay: Sleeping for %.2f seconds...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("[API Error] Failed to generate summary due to API limitations.")
                    raise e
        return ""