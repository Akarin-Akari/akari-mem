"""
Embedding providers for akari-mem-mcp.

Supports 4 modes:
1. LOCAL     — sentence-transformers (BGE-M3, etc.), best quality
2. FASTEMBED — ONNX-based FastEmbed (BGE-M3, etc.), lightweight & fast
3. API       — OpenAI-compatible embedding API, zero local resources
4. DEFAULT   — ChromaDB built-in (all-MiniLM-L6-v2), fallback
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
                    "Run: pip install sentence-transformers"
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
        # Use __call__ (batch API) — embed_query has broken behavior in v1.5.5
        raw = self._fn(texts)
        results = []
        for vec in raw:
            # Ensure pure Python list[float] for ChromaDB Rust bindings
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


class FastEmbedProvider(EmbeddingProvider):
    """
    Lightweight ONNX-based embedding via FastEmbed (by Qdrant).
    Uses quantized ONNX models — much lower memory and faster CPU inference
    compared to sentence-transformers (PyTorch). Lazy-loads on first call.

    Supported models include BAAI/bge-small-zh-v1.5, jinaai/jina-embeddings-v2-base-zh, etc.
    """

    # Known model dimensions (avoids loading model just to check dim)
    _KNOWN_DIMS = {
        "BAAI/bge-small-zh-v1.5": 512,
        "jinaai/jina-embeddings-v2-base-zh": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models/fastembed"
        )
        self._model = None
        self._dim = self._KNOWN_DIMS.get(model_name)

    def _load(self):
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self._model_name} ...")
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise RuntimeError(
                    "fastembed not installed. "
                    "Run: pip install fastembed"
                )
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
            # Probe dimension if not known
            if self._dim is None:
                probe = list(self._model.embed(["dim probe"]))
                self._dim = len(probe[0])
            logger.info(
                f"FastEmbed model loaded: {self._model_name} (dim={self._dim})"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        # FastEmbed.embed() returns a generator of numpy arrays
        return [vec.tolist() for vec in self._model.embed(texts)]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name


class ChromaEmbeddingAdapter:
    """
    Adapter that wraps EmbeddingProvider into ChromaDB's EmbeddingFunction
    protocol. Supports both v0.x (__call__) and v1.x (embed_query/embed_documents).
    """

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def __call__(self, input: List[str]) -> List[List[float]]:
        vecs = self.provider.embed(input)
        # Ensure pure Python list for ChromaDB Rust bindings
        return [v.tolist() if hasattr(v, 'tolist') else v for v in vecs]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """ChromaDB v1.x: embed documents for storage."""
        vecs = self.provider.embed(documents)
        return [v.tolist() if hasattr(v, 'tolist') else v for v in vecs]

    def embed_query(self, input: str) -> List[float]:
        """ChromaDB v1.x: embed a single query for search."""
        vec = self.provider.embed([input])[0]
        return vec.tolist() if hasattr(vec, 'tolist') else vec

    def name(self) -> str:
        """Required by ChromaDB to identify the embedding function."""
        return f"akari_mem_{self.provider.model_name}"


def create_provider(config: dict) -> EmbeddingProvider:
    """
    Factory: create embedding provider from config dict.

    Config examples:
      {"mode": "local", "model": "BAAI/bge-m3"}
      {"mode": "fastembed", "model": "BAAI/bge-small-zh-v1.5"}
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
    elif mode == "fastembed":
        return FastEmbedProvider(
            model_name=config.get("model", "BAAI/bge-small-zh-v1.5"),
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
