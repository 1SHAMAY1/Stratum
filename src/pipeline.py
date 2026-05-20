"""
Stratum Pipeline Orchestrator
================================

Connects document ingestion, RAPTOR tree construction, in-memory vector
storage, cosine similarity retrieval, and Gemini LLM answer generation.
Diagnostics are comprehensively printed out to standard output console.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import settings
from src.clustering import RaptorTreeBuilder, embedding_limiter, llm_limiter

# ──────────────────────────────────────────────────────────────────────────────
# Safe Console Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("stratum.pipeline")
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
    """Parse APIError message for specified retry delays.

    Args:
        err (APIError): Raw Google API Exception.

    Returns:
        float | None: Retry seconds from message payload if possible.
    """
    if not err.message:
        return None
    match = re.search(r"Please retry in ([\d\.]+)s", err.message)
    if match:
        return float(match.group(1))
    return None


class StratumPipeline:
    """
    Main orchestration pipeline for the Stratum RAG system.

    Attributes:
        tree_builder (RaptorTreeBuilder): Underlying RAPTOR model tree scheduler.
        all_nodes (list[dict[str, Any]]): Collected database elements across layers.
        node_embeddings (np.ndarray): Combined matrix containing vectors.
        is_ingested (bool): Flag representing model readiness.
        client (genai.Client): Instantiated Gemini Client.
    """

    def __init__(self) -> None:
        """Initialize the pipeline backend with lazy Client creation."""
        self.tree_builder = RaptorTreeBuilder()
        self.all_nodes: list[dict[str, Any]] = []
        self.node_embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self.is_ingested: bool = False
        
        # Instantiate the client lazily on initialization (NOT at import time)
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        logger.info("StratumPipeline Orchestrator initialized successfully.")

    # ── Public Methods ─────────────────────────────────────────────────────

    def ingest_raw_documents(self, raw_text: str) -> dict[str, int]:
        """
        Ingest a raw document string and build the full RAPTOR tree.

        Args:
            raw_text (str): Aggregated documents input.

        Returns:
            dict[str, int]: Performance map with item statistics.
        """
        if not raw_text.strip():
            logger.error("Attempted document ingestion with empty string.")
            raise ValueError("raw_text must not be empty.")

        logger.info("--- Initiating Knowledge Tree Ingestion ---")
        logger.info("Raw input character count: %d", len(raw_text))

        self.all_nodes = []
        self.is_ingested = False

        # Step 1: Subdivide text content
        chunks = self._chunk_text(raw_text)
        logger.info("Divided input text into %d chunks with size=%d overlap=%d", len(chunks), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        # Step 2: Vectorize basic layer
        embeddings: np.ndarray = self.tree_builder.get_embeddings(chunks)

        base_nodes: list[dict[str, Any]] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node: dict[str, Any] = {
                "id": f"base_{uuid.uuid4().hex[:8]}_{i}",
                "text": chunk,
                "embedding": emb.tolist(),
                "layer": 0,
                "is_summary": False,
            }
            base_nodes.append(node)

        logger.info("Assembled base layer containing %d nodes.", len(base_nodes))

        # Step 3: Trigger tree building steps recursively
        summary_nodes = self.tree_builder.build_tree_layer(
            base_nodes, current_layer=0
        )

        # Step 4: Aggregate and index
        self.all_nodes = base_nodes + summary_nodes
        self.node_embeddings = np.array(
            [node["embedding"] for node in self.all_nodes],
            dtype=np.float32,
        )
        self.is_ingested = True

        stats = {
            "base_nodes": len(base_nodes),
            "summary_nodes": len(summary_nodes),
            "total_nodes": len(self.all_nodes),
        }
        logger.info("--- Ingestion Complete ---")
        logger.info("Knowledge tree index details: %s", stats)
        return stats

    def execute_query(self, question: str) -> str:
        """
        Run a full RAG query with robust 429/503 retries.

        Args:
            question (str): Prompt to answer.

        Returns:
            str: Generated natural language answer.
        """
        if not self.is_ingested:
            logger.error("Attempted to execute query but no knowledge tree is loaded.")
            raise RuntimeError(
                "No documents ingested. Call ingest_raw_documents() first."
            )

        logger.info("--- Executing Pipeline Query ---")
        logger.info("Query: '%s'", question)

        # Step 1: Retrieve context vectors
        top_nodes = self._retrieve_top_k(question, k=4)
        logger.info("Retrieved %d relevant context matches from vector index.", len(top_nodes))

        summary_ctx: list[str] = []
        detail_ctx: list[str] = []

        for idx, node in enumerate(top_nodes):
            label = f"[Layer {node['layer']}]"
            logger.info("Match %d | Node ID: %s | Layer: %d | Summary: %s | Match snippet: %s...", 
                        idx + 1, node["id"], node["layer"], node["is_summary"], node["text"][:60].replace('\n', ' '))
            if node["is_summary"]:
                summary_ctx.append(f"{label} {node['text']}")
            else:
                detail_ctx.append(f"{label} {node['text']}")

        # Step 2: Build the context window block
        context_block = ""
        if summary_ctx:
            context_block += "=== HIGH-LEVEL CONTEXT ===\n"
            context_block += "\n\n".join(summary_ctx) + "\n\n"
        if detail_ctx:
            context_block += "=== DETAILED CONTEXT ===\n"
            context_block += "\n\n".join(detail_ctx)

        prompt = (
            "You are Stratum, a precise and expert technical assistant. "
            "Answer the question below based strictly on the provided context. "
            "Be thorough, accurate, and well-structured. "
            "If the context does not contain sufficient information, say so clearly.\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        logger.info("Sending prompt to LLM [%s] (Grounded context total size: %d chars)...", settings.LLM_MODEL, len(context_block))

        # Acquire 1 token from our synchronized LLM sliding window
        llm_limiter.acquire(1)

        max_retries = 6
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                t_start = time.perf_counter()
                response = self.client.models.generate_content(
                    model=settings.LLM_MODEL,
                    contents=prompt,
                )
                duration = time.perf_counter() - t_start
                logger.info("Answer generated successfully in %.2fs", duration)
                return response.text.strip()
            except APIError as e:
                logger.warning(
                    "[API Warning] LLM generation failed on attempt %d/%d (Code %d). Message: %s",
                    attempt + 1, max_retries, e.code, e.message
                )
                if e.code in (429, 503) and attempt < max_retries - 1:
                    delay = _get_retry_delay(e)
                    sleep_time = delay + 1.5 if delay is not None else base_delay * (2 ** attempt)
                    logger.info("Applying query backoff: Sleeping for %.2f seconds...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("[API Error] Failed to generate answer due to API issues.")
                    raise e
        return ""

    # ── Private Methods ────────────────────────────────────────────────────

    def _chunk_text(self, raw_text: str) -> list[str]:
        """Subdivide document text based on configured boundaries.

        Args:
            raw_text (str): Consolidated body text.

        Returns:
            list[str]: Separated slices of text.
        """
        chunks: list[str] = []
        start = 0
        text_len = len(raw_text)
        size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        while start < text_len:
            end = min(start + size, text_len)
            chunk = raw_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start += size - overlap

        return chunks

    def _retrieve_top_k(
        self, query: str, k: int = 4
    ) -> list[dict[str, Any]]:
        """Retrieve top-K nodes with robust 429/503 retries.

        Args:
            query (str): Searching question string.
            k (int): Total targets to acquire.

        Returns:
            list[dict[str, Any]]: Matched knowledge elements.
        """
        max_retries = 6
        base_delay = 2.0
        query_vec = None

        logger.info("Vectorizing search query: '%s'", query[:60])

        # Acquire 1 token from our synchronized embedding sliding window
        embedding_limiter.acquire(1)

        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                query_vec = np.array(result.embeddings[0].values, dtype=np.float32)
                break
            except APIError as e:
                logger.warning(
                    "[API Warning] Query embedding generation failed on attempt %d/%d (Code %d). Message: %s",
                    attempt + 1, max_retries, e.code, e.message
                )
                if e.code in (429, 503) and attempt < max_retries - 1:
                    delay = _get_retry_delay(e)
                    sleep_time = delay + 1.5 if delay is not None else base_delay * (2 ** attempt)
                    logger.info("Applying query backoff: Sleeping for %.2f seconds...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("[API Error] Failed to generate query vector.")
                    raise e

        if query_vec is None:
            logger.warning("Could not vectorize query. Returning empty context array.")
            return []

        # Cosine Similarity Calculation using NumPy
        node_norms = np.linalg.norm(self.node_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = node_norms * query_norm
        denom = np.where(denom == 0.0, 1e-10, denom)

        similarities = np.dot(self.node_embeddings, query_vec) / denom
        top_k_indices = np.argsort(similarities)[::-1][:k]

        retrieved = [self.all_nodes[i] for i in top_k_indices]
        # Log match scores
        for idx, i in enumerate(top_k_indices):
            logger.info("Cosine Similarity Match %d score: %.4f (Layer: %d, ID: %s)", 
                        idx + 1, similarities[i], self.all_nodes[i]["layer"], self.all_nodes[i]["id"])

        return retrieved