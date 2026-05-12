"""
Rerank module for akari-mem-mcp.

Two-stage retrieval: first retrieve candidates via embedding search,
then re-score with a cross-encoder reranker for higher precision.

Supports 4 modes:
1. LOCAL     — cross-encoder via sentence-transformers (PyTorch)
2. FASTEMBED — cross-encoder via FastEmbed ONNX (lightweight)
3. API       — Jina/Cohere rerank API
4. NONE      — disabled (default), pass-through
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger("akari-mem.rerank")


class Reranker(ABC):
    """Abstract base for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-score and re-order documents by relevance to query.

        Args:
            query: The search query
            documents: List of memory dicts (must have 'title' and 'text' keys)
            top_k: Number of top results to return

        Returns:
            Reranked list of memory dicts, with 'rerank_score' added.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LocalReranker(Reranker):
    """
    Local cross-encoder reranker via sentence-transformers.
    Recommended: BAAI/bge-reranker-v2-m3 (multilingual, SOTA)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "HF_HOME", None
        )
        self._device = device  # None = auto-detect
        self._model = None

    def _resolve_device(self) -> str:
        """Resolve device: explicit > env > auto-detect (CUDA > CPU)."""
        if self._device and self._device != "auto":
            return self._device
        env_device = os.environ.get("AKARI_DEVICE")
        if env_device:
            return env_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _load(self):
        if self._model is None:
            device = self._resolve_device()
            logger.info(f"Loading reranker model: {self._model_name} on device={device} ...")
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            self._model = CrossEncoder(
                self._model_name, cache_folder=self._cache_dir, device=device
            )
            logger.info(f"Reranker model loaded on device={device}.")

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        self._load()

        if not documents:
            return []

        # Build query-document pairs
        pairs = [
            (query, f"{doc['title']}\n{doc['text']}")
            for doc in documents
        ]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]

    @property
    def model_name(self) -> str:
        return self._model_name


class APIReranker(Reranker):
    """
    Online reranker via Jina/Cohere-compatible API.

    Jina: https://api.jina.ai/v1/rerank
    Cohere: https://api.cohere.ai/v1/rerank
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "jina-reranker-v2-base-multilingual",
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        import urllib.request

        # Build documents list for API
        doc_texts = [f"{d['title']}\n{d['text']}" for d in documents]

        payload = json.dumps({
            "model": self._model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": top_k,
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

        # API returns: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        reranked = []
        for item in result.get("results", []):
            idx = item["index"]
            doc = documents[idx].copy()
            doc["rerank_score"] = item.get("relevance_score", 0.0)
            reranked.append(doc)

        return reranked

    @property
    def model_name(self) -> str:
        return self._model_name


class FastEmbedReranker(Reranker):
    """
    Lightweight ONNX-based cross-encoder reranker via FastEmbed (by Qdrant).
    Recommended: jinaai/jina-reranker-v2-base-multilingual

    Compared to LocalReranker (PyTorch): lower memory, faster CPU inference.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models/fastembed"
        )
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info(f"Loading FastEmbed reranker: {self._model_name} ...")
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError:
                raise RuntimeError(
                    "fastembed not installed. "
                    "Run: pip install fastembed"
                )
            self._model = TextCrossEncoder(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
            logger.info("FastEmbed reranker loaded.")

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        self._load()

        if not documents:
            return []

        passages = [f"{doc['title']}\n{doc['text']}" for doc in documents]

        # TextCrossEncoder.rerank() returns an iterable of float scores
        # in the same order as the input passages
        scores = list(self._model.rerank(query, passages))

        # Attach scores and sort
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]

    @property
    def model_name(self) -> str:
        return self._model_name


class NoReranker(Reranker):
    """Pass-through: no reranking, return as-is."""

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        return documents[:top_k]

    @property
    def model_name(self) -> str:
        return "none"


def create_reranker(config: dict) -> Reranker:
    """
    Factory: create reranker from config dict.

    Config examples:
      {"mode": "none"}                                    # disabled (default)
      {"mode": "local", "model": "BAAI/bge-reranker-v2-m3"}
      {"mode": "fastembed", "model": "jinaai/jina-reranker-v2-base-multilingual"}
      {"mode": "api", "url": "https://api.jina.ai/v1/rerank",
       "key": "jina_xxx", "model": "jina-reranker-v2-base-multilingual"}
    """
    mode = config.get("mode", "none")

    if mode == "local":
        return LocalReranker(
            model_name=config.get("model", "BAAI/bge-reranker-v2-m3"),
            cache_dir=config.get("cache_dir"),
            device=config.get("device"),
        )
    elif mode == "fastembed":
        return FastEmbedReranker(
            model_name=config.get("model", "jinaai/jina-reranker-v2-base-multilingual"),
            cache_dir=config.get("cache_dir"),
        )
    elif mode == "api":
        return APIReranker(
            api_url=config["url"],
            api_key=config.get("key", ""),
            model_name=config.get("model", "jina-reranker-v2-base-multilingual"),
        )
    else:
        return NoReranker()
