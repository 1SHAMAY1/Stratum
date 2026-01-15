import numpy as np
from typing import Any
from google import genai
from google.genai import types
from config.settings import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)

class RaptorTreeBuilder:
    def __init__(self) -> None:
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

    def build_tree_layer(self, nodes: list[dict[str, Any]], current_layer: int = 0) -> list[dict[str, Any]]:
        return []
