import uuid
from typing import Any
import numpy as np
from config.settings import settings
from src.clustering import RaptorTreeBuilder

class StratumPipeline:
    def __init__(self) -> None:
        self.tree_builder = RaptorTreeBuilder()
        self.all_nodes = []
        self.node_embeddings = np.empty((0,), dtype=np.float32)
        self.is_ingested = False

    def _chunk_text(self, raw_text: str) -> list[str]:
        chunks = []
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

    def ingest_raw_documents(self, raw_text: str) -> dict[str, int]:
        chunks = self._chunk_text(raw_text)
        embeddings = self.tree_builder.get_embeddings(chunks)
        base_nodes = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node = {
                "id": f"base_{uuid.uuid4().hex[:8]}_{i}",
                "text": chunk,
                "embedding": emb.tolist(),
                "layer": 0,
                "is_summary": False,
            }
            base_nodes.append(node)
        summary_nodes = self.tree_builder.build_tree_layer(base_nodes)
        self.all_nodes = base_nodes + summary_nodes
        self.node_embeddings = np.array([n["embedding"] for n in self.all_nodes], dtype=np.float32)
        self.is_ingested = True
        return {"total_nodes": len(self.all_nodes)}

    def execute_query(self, question: str) -> str:
        return ""
