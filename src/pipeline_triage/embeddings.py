"""Text embedding models behind one small interface.

`FastEmbedEmbedder` is the real model: BAAI/bge-small-en-v1.5 run locally with
ONNX through fastembed, so there is no API key, no GPU and no PyTorch install.
The model (about 70 MB) downloads from Hugging Face on first use.

`HashingEmbedder` is a deterministic bag-of-words stand-in for tests. It needs no
download and still ranks documents by shared vocabulary, which is enough to test
the storage and search code paths.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # imported lazily: loading onnxruntime is slow

        self._model = TextEmbedding(model_name)
        self._dim = len(self.embed_query("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect queries and passages to be embedded differently.
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


class HashingEmbedder:
    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in self._token.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest) % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
