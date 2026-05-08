"""
Text chunking module for akari-mem-mcp.

Splits long text into smaller chunks for better vector search recall.
Short texts (≤ max_chars) pass through unchanged.

Strategy:
  1. Split by paragraph (\\n\\n)
  2. If a paragraph exceeds max_chars, split by sentence
  3. Merge small paragraphs until approaching max_chars
  4. Maintain overlap between chunks for context continuity

Zero external dependencies — pure Python.
"""
import re
import logging
from typing import List

logger = logging.getLogger("akari-mem.chunker")

# Sentence boundary pattern: period/question/exclamation + space,
# or Chinese sentence-ending punctuation
_SENTENCE_RE = re.compile(r'(?<=[.!?。！？；\n])\s*')


def chunk_text(
    text: str,
    max_chars: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into chunks for vector embedding.

    Args:
        text: The full text to split.
        max_chars: Maximum characters per chunk (default 500).
        overlap: Number of overlap characters between chunks (default 50).

    Returns:
        List of text chunks. Short texts return [text] as-is.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Short text — no splitting needed
    if len(text) <= max_chars:
        return [text]

    # Stage 1: Split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Stage 2: Break oversized paragraphs into sentences
    segments: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            segments.append(para)
        else:
            # Split by sentence boundaries
            sentences = _SENTENCE_RE.split(para)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                segments.extend(sentences)
            else:
                # No sentence boundaries found — hard split
                for i in range(0, len(para), max_chars - overlap):
                    segments.append(para[i : i + max_chars])

    # Stage 3: Merge small segments into chunks ≤ max_chars
    chunks: List[str] = []
    current = ""

    for seg in segments:
        candidate = f"{current}\n\n{seg}" if current else seg
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If this single segment exceeds max_chars, it was already
            # split in Stage 2, so just add it directly
            current = seg

    if current:
        chunks.append(current)

    # Stage 4: Add overlap — prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}...{chunks[i]}")
        chunks = overlapped

    logger.debug(f"Chunked text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks
