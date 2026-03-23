"""
Embedding providers for akari-mem-mcp.

Supports 3 modes:
1. LOCAL  — sentence-transformers (BGE-M3, etc.), best quality
2. API    — OpenAI-compatible embedding API, zero local resources
3. DEFAULT — ChromaDB built-in (all-MiniLM-L6-v2), fallback
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger("akari-mem.embeddings")


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding via sentence-transformers. Lazy-loads on first call."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models"
        )
        self._model = None
        self._dim: Optional[int] = None

    def _load(self):
        if self._model is None:
            logger.info(f"Loading local model: {self._model_name} ...")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers --target F:\\python-libs"
                )
            self._model = SentenceTransformer(
                self._model_name, cache_folder=self._cache_dir
            )
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded: dim={self._dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name


class APIEmbeddingProvider(EmbeddingProvider):
    """
    Online embedding via OpenAI-compatible API.
    Works with: OpenAI, Azure OpenAI, Ollama, LiteLLM, vLLM, etc.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        import urllib.request

        payload = json.dumps({
            "model": self._model_name,
            "input": texts,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class DefaultEmbeddingProvider(EmbeddingProvider):
    """ChromaDB built-in (all-MiniLM-L6-v2). Zero-config fallback."""

    def __init__(self):
        self._fn = None
        self._dim = 384

    def _load(self):
        if self._fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    DefaultEmbeddingFunction,
                )
                self._fn = DefaultEmbeddingFunction()
            except ImportError:
                from chromadb.utils.embedding_functions import (
                    ONNXMiniLM_L6_V2,
                )
                self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        # ChromaDB v1.5.5: embed_query(str) -> List[float]
        results = []
        for text in texts:
            vec = self._fn.embed_query(text)
            if hasattr(vec, 'tolist'):
                vec = vec.tolist()
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2"


class ChromaEmbeddingAdapter:
    """
    Adapter that wraps EmbeddingProvider into ChromaDB's EmbeddingFunction
    protocol. Supports both v0.x (__call__) and v1.x (embed_query/embed_documents).
    """

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.provider.embed(input)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """ChromaDB v1.x: embed documents for storage."""
        return self.provider.embed(documents)

    def embed_query(self, input: str) -> List[float]:
        """ChromaDB v1.x: embed a single query for search."""
        return self.provider.embed([input])[0]

    def name(self) -> str:
        """Required by ChromaDB to identify the embedding function."""
        return f"akari_mem_{self.provider.model_name}"


def create_provider(config: dict) -> EmbeddingProvider:
    """
    Factory: create embedding provider from config dict.

    Config examples:
      {"mode": "local", "model": "BAAI/bge-m3"}
      {"mode": "api", "url": "https://api.openai.com/v1/embeddings",
       "key": "sk-...", "model": "text-embedding-3-small", "dim": 1536}
      {"mode": "default"}
    """
    mode = config.get("mode", "default")

    if mode == "local":
        return LocalEmbeddingProvider(
            model_name=config.get("model", "BAAI/bge-m3"),
            cache_dir=config.get("cache_dir"),
        )
    elif mode == "api":
        return APIEmbeddingProvider(
            api_url=config["url"],
            api_key=config.get("key", ""),
            model_name=config.get("model", "text-embedding-3-small"),
            dimension=config.get("dim", 1536),
        )
    else:
        return DefaultEmbeddingProvider()
