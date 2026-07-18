"""
BM25 Index — rank_bm25 wrapper for sparse retrieval.

See docs/2_system_architecture.md §6 for spec.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging
import pickle
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Regex that splits on whitespace and common C++ punctuation / operators.
_SPLIT_RE = re.compile(r"[\s()\{\};,<>*&:./]+|->|::")

# Regex for CamelCase boundary detection.
# Matches transitions like "Str|Cat", "Make|Span", "BM25|Okapi"
_CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def _split_camel(token: str) -> List[str]:
    """Split a CamelCase token into sub-parts.

    Examples:
        "StrCat"     → ["str", "cat"]
        "MakeSpan"   → ["make", "span"]
        "getNode"    → ["get", "node"]
        "BM25Okapi"  → ["bm", "25", "okapi"]
        "str_cat"    → ["str_cat"]  (no CamelCase, returned as-is)
    """
    parts = _CAMEL_RE.sub(' ', token).split()
    return [p.lower() for p in parts if len(p) >= 2]


class BM25Index:
    """Sparse BM25 retrieval over code chunk texts."""

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._chunks: List = []   # parallel list to BM25 corpus
        self._symbol_lookup: Dict[str, set[int]] = {}
        self._file_lookup: Dict[str, set[int]] = {}

    def build(self, chunks: List) -> None:
        """Build BM25 index from a list of Chunk objects.

        Tokenizes code_text by whitespace + C++ operator stripping.
        Preserves identifier names (CamelCase, snake_case) intact.

        Args:
            chunks: List of Chunk dataclass objects.
        """
        self._chunks = list(chunks)
        corpus = [self._tokenize(self._build_search_text(c)) for c in self._chunks]
        self._index = BM25Okapi(corpus)
        self._symbol_lookup = self._build_symbol_lookup(self._chunks)
        self._file_lookup = self._build_file_lookup(self._chunks)
        logger.info("Built BM25 index with %d chunks.", len(self._chunks))

    def query(self, text: str, top_k: int = 20) -> List[Dict]:
        """Score all chunks against query text.

        Args:
            text: Query string (log error_message + identifiers).
            top_k: Number of results to return.

        Returns:
            List of dicts: {chunk_id, file_path, function_name,
            start_line, score}.
            Score is raw BM25 score (not normalised).
            Returns empty list if all scores are zero.
        """
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(text)
        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        # If every score is zero there are no useful results.
        if np.allclose(scores, 0.0):
            return []

        # Argsort descending, take top_k.
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "symbol_name": getattr(chunk, "symbol_name", ""),
                "class_name": getattr(chunk, "class_name", ""),
                "namespace": getattr(chunk, "namespace", ""),
                "signature": getattr(chunk, "signature", ""),
                "start_line": chunk.start_line,
                "score": score,
            })

        return results

    def query_by_symbols(
        self,
        identifiers: List[str],
        top_k: int = 20,
    ) -> List[Dict]:
        """Return chunks matched by symbol names extracted from logs."""
        if not identifiers or not self._chunks:
            return []

        score_by_idx: Dict[int, float] = defaultdict(float)

        for raw_ident in identifiers:
            ident = self._normalize_symbol(raw_ident)
            if not ident:
                continue

            base = ident.split("::")[-1]
            search_terms = [(ident, 3.0)]
            if base != ident:
                search_terms.append((base, 1.75))

            for term, weight in search_terms:
                for idx in self._symbol_lookup.get(term, set()):
                    score_by_idx[idx] += weight

        if not score_by_idx:
            return []

        ranked = sorted(
            score_by_idx.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Dict] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "symbol_name": getattr(chunk, "symbol_name", ""),
                "class_name": getattr(chunk, "class_name", ""),
                "namespace": getattr(chunk, "namespace", ""),
                "signature": getattr(chunk, "signature", ""),
                "start_line": chunk.start_line,
                "score": float(score),
            })
        return results

    def query_by_path_prefix(
        self,
        prefixes: List[str],
        top_k: int = 20,
    ) -> List[Dict]:
        """Return chunks whose paths share important directory prefixes."""
        if not prefixes or not self._chunks:
            return []

        normalized_prefixes = []
        for prefix in prefixes:
            norm = str(prefix or "").replace("\\", "/").strip("/")
            if norm and norm not in normalized_prefixes:
                normalized_prefixes.append(norm)

        if not normalized_prefixes:
            return []

        score_by_idx: Dict[int, float] = defaultdict(float)
        for idx, chunk in enumerate(self._chunks):
            file_path = str(getattr(chunk, "file_path", "")).replace("\\", "/")
            for prefix in normalized_prefixes:
                if file_path == prefix or file_path.startswith(f"{prefix}/"):
                    # Longer/more specific prefixes are weighted higher.
                    specificity = min(len(prefix.split("/")), 6)
                    score_by_idx[idx] = max(score_by_idx[idx], 1.5 + 0.25 * specificity)

        if not score_by_idx:
            return []

        ranked = sorted(
            score_by_idx.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Dict] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "symbol_name": getattr(chunk, "symbol_name", ""),
                "class_name": getattr(chunk, "class_name", ""),
                "namespace": getattr(chunk, "namespace", ""),
                "signature": getattr(chunk, "signature", ""),
                "start_line": chunk.start_line,
                "score": float(score),
            })
        return results

    def query_by_file_hints(
        self,
        hints: List[str],
        top_k: int = 20,
    ) -> List[Dict]:
        """Return chunks matched by filename/stem hints (pathless-safe)."""
        if not hints or not self._chunks:
            return []

        score_by_idx: Dict[int, float] = defaultdict(float)
        for raw_hint in hints:
            hint = self._normalize_file_hint(raw_hint)
            if not hint:
                continue
            stem = hint.rsplit(".", 1)[0] if "." in hint else hint
            candidates = [(hint, 2.8)]
            if stem and stem != hint:
                candidates.append((stem, 1.9))
            for term, weight in candidates:
                for idx in self._file_lookup.get(term, set()):
                    score_by_idx[idx] += weight

        if not score_by_idx:
            return []

        ranked = sorted(
            score_by_idx.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Dict] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "symbol_name": getattr(chunk, "symbol_name", ""),
                "class_name": getattr(chunk, "class_name", ""),
                "namespace": getattr(chunk, "namespace", ""),
                "signature": getattr(chunk, "signature", ""),
                "start_line": chunk.start_line,
                "score": float(score),
            })
        return results

    def save(self, path: Path) -> None:
        """Persist index to disk using pickle.

        Args:
            path: File path for the pickle output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self._index,
                    "chunks": self._chunks,
                    "symbol_lookup": self._symbol_lookup,
                    "file_lookup": self._file_lookup,
                },
                f,
            )
        logger.info("Saved BM25 index to %s.", path)

    def load(self, path: Path) -> None:
        """Load index from disk.

        Args:
            path: File path to the pickled index.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._chunks = data["chunks"]
        self._symbol_lookup = data.get("symbol_lookup") or self._build_symbol_lookup(self._chunks)
        self._file_lookup = data.get("file_lookup") or self._build_file_lookup(self._chunks)
        logger.info(
            "Loaded BM25 index from %s (%d chunks).", path, len(self._chunks)
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize code text for BM25.

        Splits on whitespace and common C++ operators/punctuation.
        Filters out tokens shorter than 2 characters and pure-numeric
        tokens.  Lowercases everything.  Also splits CamelCase tokens
        into sub-parts so that ``StrCat`` matches ``str_cat``.

        Args:
            text: Raw code or query text.

        Returns:
            List of lowercase token strings.
        """
        raw_tokens = _SPLIT_RE.split(text)
        tokens: List[str] = []
        for tok in raw_tokens:
            tok = tok.strip()
            if len(tok) < 2:
                continue
            lowered = tok.lower()
            if lowered.isdigit():
                continue

            # Always emit the full lowered token.
            tokens.append(lowered)

            # Also emit CamelCase sub-parts if present.
            camel_parts = _split_camel(tok)
            if len(camel_parts) > 1:
                tokens.extend(camel_parts)

            # Also split snake_case tokens (e.g. str_cat → [str, cat]).
            if '_' in lowered:
                snake_parts = [p for p in lowered.split('_') if len(p) >= 2]
                if len(snake_parts) > 1:
                    tokens.extend(snake_parts)

        return tokens

    def debug_tokenize(self, text: str) -> List[str]:
        """Public wrapper around _tokenize for debugging.

        Useful for inspecting how identifiers and code text are split.
        """
        return self._tokenize(text)

    def _build_search_text(self, chunk) -> str:
        """Build an augmented retrieval document for a chunk."""
        parts = [
            getattr(chunk, "function_name", ""),
            getattr(chunk, "symbol_name", ""),
            getattr(chunk, "class_name", ""),
            getattr(chunk, "namespace", ""),
            getattr(chunk, "signature", ""),
            getattr(chunk, "file_path", ""),
            getattr(chunk, "code_text", ""),
        ]
        return "\n".join(part for part in parts if part)

    def _build_symbol_lookup(self, chunks: List) -> Dict[str, set[int]]:
        """Build a symbol -> chunk-index lookup table."""
        lookup: Dict[str, set[int]] = defaultdict(set)
        for idx, chunk in enumerate(chunks):
            function_name = self._normalize_symbol(getattr(chunk, "function_name", ""))
            symbol_name = self._normalize_symbol(getattr(chunk, "symbol_name", ""))
            class_name = self._normalize_symbol(getattr(chunk, "class_name", ""))

            candidates = {
                function_name,
                symbol_name,
                function_name.split("::")[-1] if function_name else "",
                f"{class_name}::{symbol_name}" if class_name and symbol_name else "",
            }
            for candidate in candidates:
                if candidate:
                    lookup[candidate].add(idx)
        return dict(lookup)

    def _build_file_lookup(self, chunks: List) -> Dict[str, set[int]]:
        """Build filename/stem -> chunk-index lookup for pathless hints."""
        lookup: Dict[str, set[int]] = defaultdict(set)
        for idx, chunk in enumerate(chunks):
            file_path = str(getattr(chunk, "file_path", "")).replace("\\", "/")
            if not file_path:
                continue
            base = file_path.rsplit("/", 1)[-1].lower()
            stem = base.rsplit(".", 1)[0] if "." in base else base
            if base:
                lookup[base].add(idx)
            if stem:
                lookup[stem].add(idx)
        return dict(lookup)

    def _normalize_file_hint(self, hint: str) -> str:
        """Normalize file hint text into lookup-compatible basename."""
        value = (hint or "").strip().lower().strip("`'\"")
        value = value.replace("\\", "/")
        value = value.rsplit("/", 1)[-1]
        return value

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize raw identifier text into a lookup key."""
        value = (symbol or "").strip().lower().strip("`'\"")
        if "(" in value:
            value = value.split("(", 1)[0]
        return value.strip()
