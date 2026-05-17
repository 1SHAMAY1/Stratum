"""
Stratum Pipeline Orchestrator
================================

Connects document ingestion, RAPTOR tree construction, in-memory vector
storage, cosine similarity retrieval, and Gemini LLM answer generation.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import settings
from src.clustering import RaptorTreeBuilder

# Configure Gemini API using the global settings singleton.
_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _get_retry_delay(err: APIError) -> float | None:
    """Parse APIError message for specified retry delays."""
    if not err.message:
        return None
    match = re.search(r"Please retry in ([\d\.]+)s", err.message)
    if match:
        return float(match.group(1))
    return None


class StratumPipeline:
    """
    Main orchestration pipeline for the Stratum RAG system.
    """

    def __init__(self) -> None:
        self.tree_builder = RaptorTreeBuilder()
        self.all_nodes: list[dict[str, Any]] = []
        self.node_embeddings: np.ndarray = np.empty((0,), dtype=np.float32)
        self.is_ingested: bool = False

    # ── Public Methods ─────────────────────────────────────────────────────

    def ingest_raw_documents(self, raw_text: str) -> dict[str, int]:
        """Ingest a raw document string and build the full RAPTOR tree."""
        if not raw_text.strip():
            raise ValueError("raw_text must not be empty.")

        self.all_nodes = []
        self.is_ingested = False

        chunks = self._chunk_text(raw_text)
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

        summary_nodes = self.tree_builder.build_tree_layer(
            base_nodes, current_layer=0
        )

        self.all_nodes = base_nodes + summary_nodes
        self.node_embeddings = np.array(
            [node["embedding"] for node in self.all_nodes],
            dtype=np.float32,
        )
        self.is_ingested = True

        return {
            "base_nodes": len(base_nodes),
            "summary_nodes": len(summary_nodes),
            "total_nodes": len(self.all_nodes),
        }

    def execute_query(self, question: str) -> str:
        """Run a full RAG query with robust 429/503 retries."""
        if not self.is_ingested:
            raise RuntimeError(
                "No documents ingested. Call ingest_raw_documents() first."
            )

        top_nodes = self._retrieve_top_k(question, k=4)

        summary_ctx: list[str] = []
        detail_ctx: list[str] = []

        for node in top_nodes:
            label = f"[Layer {node['layer']}]"
            if node["is_summary"]:
                summary_ctx.append(f"{label} {node['text']}")
            else:
                detail_ctx.append(f"{label} {node['text']}")

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

        max_retries = 6
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = _client.models.generate_content(
                    model=settings.LLM_MODEL,
                    contents=prompt,
                )
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

    # ── Private Methods ────────────────────────────────────────────────────

    def _chunk_text(self, raw_text: str) -> list[str]:
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
        """Retrieve top-K nodes with robust 429/503 retries."""
        max_retries = 6
        base_delay = 2.0
        query_vec = None

        for attempt in range(max_retries):
            try:
                result = _client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                query_vec = np.array(result.embeddings[0].values, dtype=np.float32)
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

        if query_vec is None:
            return []

        node_norms = np.linalg.norm(self.node_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = node_norms * query_norm
        denom = np.where(denom == 0.0, 1e-10, denom)

        similarities = np.dot(self.node_embeddings, query_vec) / denom
        top_k_indices = np.argsort(similarities)[::-1][:k]

        return [self.all_nodes[i] for i in top_k_indices]
