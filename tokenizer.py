# -*- coding: utf-8 -*-
"""
FTS5 tokenizer based on jieba for CJK (Chinese/Japanese/Korean) support.

Strategy: Application-level pre-tokenization.
- On write: jieba.cut(text) → space-joined → stored in FTS5 index
- On query: jieba.cut(query) → space-joined → FTS5 MATCH
- FTS5 keeps default unicode61 tokenizer (splits on spaces)

This gives best-in-class Chinese segmentation without C extensions.
"""
import re
import logging

logger = logging.getLogger("akari-mem.tokenizer")

_jieba = None
_jieba_initialized = False


def _ensure_jieba():
    """Lazy-load jieba on first use. Thread-safe via GIL."""
    global _jieba, _jieba_initialized
    if _jieba_initialized:
        return _jieba
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)  # suppress jieba's verbose init logs
        jieba.initialize()  # pre-build dictionary (blocks ~1-2s first time)
        _jieba = jieba
        _jieba_initialized = True
        logger.info("jieba tokenizer initialized successfully")
    except ImportError:
        _jieba = None
        _jieba_initialized = True
        logger.warning(
            "jieba not installed — FTS5 will use raw text (Chinese search degraded). "
            "Install with: pip install jieba"
        )
    return _jieba


# Regex to detect CJK characters
_CJK_RANGES = re.compile(
    r"[\u4e00-\u9fff"       # CJK Unified Ideographs
    r"\u3400-\u4dbf"        # CJK Extension A
    r"\u3000-\u303f"        # CJK Symbols and Punctuation
    r"\uff00-\uffef"        # Fullwidth Forms
    r"\u3040-\u309f"        # Hiragana
    r"\u30a0-\u30ff"        # Katakana
    r"\uac00-\ud7af]"       # Korean Hangul
)


def has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return bool(_CJK_RANGES.search(text))


def tokenize_for_fts(text: str) -> str:
    """
    Tokenize text for FTS5 indexing.

    - If text contains CJK characters and jieba is available:
      segments with jieba, returns space-joined tokens.
    - Otherwise: returns original text (unicode61 handles it fine).

    Always strips excess whitespace and normalizes.
    """
    if not text:
        return ""

    jieba = _ensure_jieba()

    if jieba is None or not has_cjk(text):
        # No jieba or pure ASCII/Latin — return as-is
        return text.strip()

    # jieba.cut returns generator of tokens
    # Use cut_for_search for finer granularity (better recall)
    tokens = jieba.cut_for_search(text)
    # Filter out pure whitespace/punctuation tokens, keep meaningful ones
    result_tokens = []
    for token in tokens:
        t = token.strip()
        if t and len(t) > 0:
            result_tokens.append(t)

    return " ".join(result_tokens)


def tokenize_query(query: str) -> str:
    """
    Tokenize a search query for FTS5 MATCH.

    Similar to tokenize_for_fts but uses jieba.cut (precise mode)
    for more accurate query term extraction.
    """
    if not query:
        return ""

    jieba = _ensure_jieba()

    if jieba is None or not has_cjk(query):
        return query.strip()

    # For queries, use precise mode (not search mode) to avoid over-splitting
    tokens = jieba.cut(query, cut_all=False)
    result_tokens = []
    for token in tokens:
        t = token.strip()
        if t and len(t) > 0:
            result_tokens.append(t)

    return " ".join(result_tokens)
