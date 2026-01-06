import numpy as np
from typing import Any

class RaptorTreeBuilder:
    def __init__(self) -> None:
        pass

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return np.empty((0, 3072), dtype=np.float32)

    def build_tree_layer(self, nodes: list[dict[str, Any]], current_layer: int = 0) -> list[dict[str, Any]]:
        return []
