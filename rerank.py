"""
Rerank module for akari-mem-mcp.

Two-stage retrieval: first retrieve candidates via embedding search,
then re-score with a cross-encoder reranker for higher precision.

Supports 5 modes:
1. LOCAL     — cross-encoder via sentence-transformers (PyTorch)
2. FASTEMBED — cross-encoder via FastEmbed ONNX (lightweight, no VRAM control)
3. ONNX      — custom onnxruntime cross-encoder with tight CUDA arena (preferred GPU path)
4. API       — Jina/Cohere rerank API
5. NONE      — disabled (default), pass-through
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

    IMPORTANT: FastEmbed's TextCrossEncoder defaults to `cuda=Device.AUTO`,
    which silently picks CUDA when a GPU is available. That would load the
    Jina reranker onto the same GPU as the BGE-M3 ONNX session and balloon
    VRAM by 2-3 GB per search. We force CPU here unless explicitly overridden
    via `device='cuda'` in config.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models/fastembed"
        )
        self._device = (device or "cpu").lower()
        self._model = None

    def _load(self):
        if self._model is None:
            def _tr(msg: str) -> None:
                try:
                    import time as _t, os as _os
                    _p = _os.path.join(
                        _os.path.dirname(_os.path.abspath(__file__)), "data", "warmup.log"
                    )
                    with open(_p, "a", encoding="utf-8") as _f:
                        _f.write(f"[{_t.strftime('%H:%M:%S')}] [rerank-load] {msg}\n")
                        _f.flush()
                except Exception:
                    pass

            use_cuda = self._device == "cuda"
            _tr(f"start (model={self._model_name}, cuda={use_cuda})")
            logger.info(
                f"Loading FastEmbed reranker: {self._model_name} "
                f"on device={'cuda' if use_cuda else 'cpu'} ..."
            )
            _tr("import fastembed.TextCrossEncoder")
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError:
                raise RuntimeError(
                    "fastembed not installed. "
                    "Run: pip install fastembed"
                )
            _tr("TextCrossEncoder() start (may probe network or unpack ONNX)")
            self._model = TextCrossEncoder(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                cuda=use_cuda,  # ← pin to CPU by default; opt-in for CUDA
            )
            _tr("TextCrossEncoder() done")
            # Drop protobuf temp buffer; protects subsequent session loads.
            import gc as _gc
            _gc.collect()
            logger.info(
                f"FastEmbed reranker loaded (device={'cuda' if use_cuda else 'cpu'})."
            )

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

    # ── Idle-unload API (swap-style; see embeddings.py for rationale) ──
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        import gc
        if self._model is None:
            return
        logger.info(f"Unloading FastEmbed reranker ({self._model_name})...")
        self._model = None
        gc.collect()


class OnnxReranker(Reranker):
    """
    Custom ONNX cross-encoder reranker bypassing fastembed's TextCrossEncoder.

    Why this exists:
    - fastembed.TextCrossEncoder does NOT expose ORT SessionOptions or
      provider_options. That means we cannot cap the CUDA arena or disable
      the cuDNN max-workspace probing. On GPU that lets VRAM creep to
      several GB per inference batch.
    - This class loads the SAME ONNX file fastembed already downloaded into
      its cache (default F:/models/fastembed/...) but builds the ORT session
      with the exact tight-arena config used by OnnxEmbeddingProvider:
        arena_extend_strategy=kSameAsRequested,
        cudnn_conv_use_max_workspace=0,
        cudnn_conv_algo_search=HEURISTIC,
        gpu_mem_limit cap (default 1 GB — jina-reranker-v2-base FP32 ~1.1GB
        weights + ~0.4GB activations comfortably fits under 1.5GB after the
        arena cap; if VRAM still spikes raise to 1.5 in config.json).
    - Uses tokenizers.Tokenizer.from_file (Rust) instead of transformers
      AutoTokenizer for the same MCP-stdio-blocking reason as BGE-M3.
    - Truncates pairs to max_length=256 by default (vs FastEmbed's 512) for
      ~2x rerank throughput at negligible quality loss for short snippets.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 256,
        gpu_mem_limit_gb: float = 1.0,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models/fastembed"
        )
        self._device = (device or "cuda").lower()
        self._max_length = int(max_length)
        self._gpu_mem_limit_bytes = int(gpu_mem_limit_gb * 1024 * 1024 * 1024)
        self._session = None
        self._tokenizer = None
        self._output_name: Optional[str] = None
        self._needs_token_type_ids: bool = False

    def _find_artifacts(self):
        """Locate ONNX + tokenizer in fastembed's HuggingFace-style cache.

        Layout:
            <cache>/models--{org}--{name}/snapshots/<hash>/onnx/model.onnx
            <cache>/models--{org}--{name}/snapshots/<hash>/onnx/model_fp16.onnx  (optional)
            <cache>/models--{org}--{name}/snapshots/<hash>/tokenizer.json

        Precedence: model_fp16.onnx > model.onnx
        (FP16 cuts weights from ~1.1 GB → 531 MB, ~50% VRAM savings.)
        """
        repo_dir = "models--" + self._model_name.replace("/", "--")
        snap_root = os.path.join(self._cache_dir, repo_dir, "snapshots")
        if not os.path.isdir(snap_root):
            raise FileNotFoundError(
                f"Reranker cache not found: {snap_root}. "
                f"Run once with mode=fastembed to populate the cache, "
                f"or set rerank.cache_dir in config.json."
            )
        snaps = [
            d for d in os.listdir(snap_root)
            if os.path.isdir(os.path.join(snap_root, d))
        ]
        if not snaps:
            raise FileNotFoundError(f"No snapshots under {snap_root}")
        snap_dir = os.path.join(snap_root, snaps[0])
        onnx_dir = os.path.join(snap_dir, "onnx")
        # Prefer FP16 if present (half VRAM, half I/O).
        fp16_file = os.path.join(onnx_dir, "model_fp16.onnx")
        fp32_file = os.path.join(onnx_dir, "model.onnx")
        if os.path.exists(fp16_file):
            onnx_file = fp16_file
        elif os.path.exists(fp32_file):
            onnx_file = fp32_file
        else:
            raise FileNotFoundError(
                f"Neither model_fp16.onnx nor model.onnx found in {onnx_dir}"
            )
        tok_file = os.path.join(snap_dir, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(f"tokenizer.json not found: {tok_file}")
        return onnx_file, tok_file

    def _load(self):
        if self._session is not None:
            return

        def _tr(msg: str) -> None:
            try:
                import time as _t
                _p = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data", "warmup.log",
                )
                with open(_p, "a", encoding="utf-8") as _f:
                    _f.write(f"[{_t.strftime('%H:%M:%S')}] [onnx-rerank-load] {msg}\n")
                    _f.flush()
            except Exception:
                pass

        _tr(f"start (model={self._model_name}, device={self._device})")

        import onnxruntime as ort
        ort.set_default_logger_severity(3)

        onnx_file, tok_file = self._find_artifacts()
        _tr(f"artifacts onnx={onnx_file}")

        from tokenizers import Tokenizer as _FastTok
        self._tokenizer = _FastTok.from_file(tok_file)
        # XLM-RoBERTa family (jina-reranker-v2 inherits): pad_id=1 (<pad>).
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=self._max_length)
        _tr("tokenizer ready")

        # Tight CUDA arena (matches OnnxEmbeddingProvider).
        providers = []
        if self._device == "cuda":
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                    "do_copy_in_default_stream": True,
                    "gpu_mem_limit": str(self._gpu_mem_limit_bytes),
                },
            ))
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        # Same EXTENDED workaround as BGE-M3 (ORT 1.23 ENABLE_ALL fusion bug).
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        sess_options.log_severity_level = 3

        # Redirect fd 2 during session build (MCP stdio pipe safety).
        _stderr_log = os.path.join(os.path.dirname(onnx_file), "load_stderr.log")
        _saved_fd2 = None
        try:
            try:
                _saved_fd2 = os.dup(2)
                _redir_fd = os.open(_stderr_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
                os.dup2(_redir_fd, 2)
                os.close(_redir_fd)
            except Exception:
                _saved_fd2 = None
            _tr("InferenceSession start")
            self._session = ort.InferenceSession(
                onnx_file, sess_options=sess_options, providers=providers
            )
            _tr("InferenceSession done")
        finally:
            if _saved_fd2 is not None:
                try:
                    os.dup2(_saved_fd2, 2)
                    os.close(_saved_fd2)
                except Exception:
                    pass

        input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in input_names

        outputs = self._session.get_outputs()
        self._output_name = outputs[0].name

        actual = self._session.get_providers()
        logger.info(
            f"OnnxReranker ready: model={self._model_name}, "
            f"providers={actual}, output='{self._output_name}', "
            f"token_type_ids={self._needs_token_type_ids}, "
            f"max_length={self._max_length}"
        )

        import gc as _gc
        _gc.collect()

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        import numpy as np
        self._load()

        if not documents:
            return []

        # Build (query, passage) pairs — tokenizers handles the special-token
        # insertion (XLM-R: <s> q </s></s> p </s>) for pair input.
        passages = [f"{doc['title']}\n{doc['text']}" for doc in documents]
        pairs = [(query, p) for p in passages]

        encs = self._tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._needs_token_type_ids:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run([self._output_name], feed)
        logits = outputs[0]

        # Cross-encoder convention: scalar logit per pair (shape [B, 1]) is
        # the relevance score. Some 2-class heads output [neg, pos]; take
        # the last column ("pos") in that case.
        if logits.ndim == 2:
            if logits.shape[1] == 1:
                scores = logits[:, 0]
            else:
                scores = logits[:, -1]
        else:
            scores = logits.ravel()

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── Idle-unload API (swap-style; see embeddings.py for rationale) ──
    def is_loaded(self) -> bool:
        return self._session is not None

    def unload(self) -> None:
        import gc
        if self._session is None:
            return
        logger.info(f"Unloading OnnxReranker ({self._model_name})...")
        self._session = None
        self._tokenizer = None
        self._output_name = None
        gc.collect()


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
            device=config.get("device", "cpu"),
        )
    elif mode == "onnx":
        return OnnxReranker(
            model_name=config.get("model", "jinaai/jina-reranker-v2-base-multilingual"),
            cache_dir=config.get("cache_dir"),
            device=config.get("device", "cuda"),
            max_length=int(config.get("max_length", 256)),
            gpu_mem_limit_gb=float(config.get("gpu_mem_limit_gb", 1.0)),
        )
    elif mode == "api":
        return APIReranker(
            api_url=config["url"],
            api_key=config.get("key", ""),
            model_name=config.get("model", "jina-reranker-v2-base-multilingual"),
        )
    else:
        return NoReranker()
