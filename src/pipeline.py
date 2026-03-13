from typing import Any
import numpy as np
from config.settings import settings

class StratumPipeline:
    def __init__(self) -> None:
        pass

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
        return {}

    def execute_query(self, question: str) -> str:
        return ""
