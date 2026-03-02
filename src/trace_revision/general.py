from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

# Shared regex tokenizer (words w/ apostrophes + punctuation as tokens)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]")


# ============================================================
# Tokenization (shared)
# ============================================================

def tokenize_regex(text: str, *, lowercase: bool = False) -> List[str]:
    """Regex tokenizer used across TRACE (punctuation preserved as tokens)."""
    if lowercase:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]
    return [m.group(0) for m in TOKEN_RE.finditer(text or "")]


@dataclass(frozen=True)
class Tok:
    """Token with inclusive character span in its local string."""
    text: str
    start: int  # inclusive
    end: int    # inclusive


def tokenize_with_spans(text: str) -> List[Tok]:
    """Tokenize using TOKEN_RE and return tokens with inclusive local char spans."""
    out: List[Tok] = []
    for m in TOKEN_RE.finditer(text or ""):
        s, e = m.span()  # e exclusive
        out.append(Tok(m.group(0), s, e - 1))
    return out


# ============================================================
# Levenshtein distance + normalized similarity (shared)
# ============================================================

def levenshtein_distance_tokens(a: List[str], b: List[str]) -> int:
    """
    Token-level Levenshtein distance (unit costs), O(min(m,n)) space.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    # Keep n <= m for memory
    if n > m:
        a, b = b, a
        m, n = n, m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + cost  # substitute/match
            )
        prev = cur
    return prev[n]


def normalized_edit_similarity_from_tokens(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Similarity in [0,1] = 1 - dist(tokens_a, tokens_b) / max(len_a, len_b).
    """
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    dist = levenshtein_distance_tokens(tokens_a, tokens_b)
    return 1.0 - dist / max(len(tokens_a), len(tokens_b))


def normalized_edit_similarity(text_a: str, text_b: str, *, lowercase: bool = True) -> float:
    """Normalized token-level similarity computed from regex tokenization."""
    a = tokenize_regex(text_a, lowercase=lowercase)
    b = tokenize_regex(text_b, lowercase=lowercase)
    return normalized_edit_similarity_from_tokens(a, b)


# ============================================================
# Span + dataframe utilities (shared)
# ============================================================

def safe_cell(v: Any) -> Any:
    """
    Safe getter for pandas cells:
      - None stays None
      - pandas/NumPy missing values -> None
      - lists/tuples/dicts are returned as-is (avoid pd.isna(list) ambiguity)
    """
    if v is None:
        return None

    # Avoid pd.isna on containers (it can return arrays / raise ambiguity)
    if isinstance(v, (list, tuple, dict)):
        return v

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    return v


_SPAN_STR_RE = re.compile(
    r"^\s*[\[\(\{]?\s*(-?\d+)\s*[,:\-\s]\s*(-?\d+)\s*[\]\)\}]?\s*$"
)

def ensure_span_list(x: Any) -> Optional[List[int]]:
    """
    Enforce Python list span format: [start, end] (ints) or None.

    Accepts:
      - None / NaN / pd.NA -> None
      - list/tuple len 2 -> [int, int]
      - numpy array / sequence-like len 2 -> [int, int]
      - string forms: "[12, 34]", "(12,34)", "12,34", "12 34", "12-34", "12:34"

    Rejects:
      - arbitrary strings (e.g., "12" will NOT become [1,2])
      - spans with invalid order or negative indices
    """
    x = safe_cell(x)
    if x is None:
        return None

    # --- Handle strings explicitly (do NOT treat as iterable) ---
    if isinstance(x, (str, bytes)):
        s = x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        s = s.strip()
        if s == "" or s.lower() in {"none", "nan", "null"}:
            return None

        m = _SPAN_STR_RE.match(s)
        if not m:
            raise ValueError(f"Invalid span string (expected '[start,end]' or 'start,end'): {x!r}")

        start = int(m.group(1))
        end = int(m.group(2))

        if start < 0 or end < 0:
            raise ValueError(f"Invalid span (negative indices): {x!r}")
        if start > end:
            raise ValueError(f"Invalid span (start > end): {x!r}")

        return [start, end]

    # --- Normal list/tuple case ---
    if isinstance(x, (list, tuple)):
        if len(x) != 2:
            raise ValueError(f"Invalid span (expected length 2): {x!r}")
        start, end = int(x[0]), int(x[1])

        if start < 0 or end < 0:
            raise ValueError(f"Invalid span (negative indices): {x!r}")
        if start > end:
            raise ValueError(f"Invalid span (start > end): {x!r}")

        return [start, end]

    # --- Other sequence-likes (e.g., numpy arrays) ---
    try:
        xs = list(x)
    except Exception as e:
        raise ValueError(f"Invalid span format (expected [start,end] or None): {x!r}") from e

    if len(xs) != 2:
        raise ValueError(f"Invalid span (expected length 2): {x!r}")

    start, end = int(xs[0]), int(xs[1])

    if start < 0 or end < 0:
        raise ValueError(f"Invalid span (negative indices): {x!r}")
    if start > end:
        raise ValueError(f"Invalid span (start > end): {x!r}")

    return [start, end]


def span_base(span: Optional[List[int]]) -> Optional[int]:
    """Return start index (base) from [start,end] span."""
    if isinstance(span, list) and len(span) == 2:
        return span[0]
    return None


def add_span_base(local_span: Optional[List[int]], base_start: Optional[int]) -> Optional[List[int]]:
    """Shift a local [start,end] span into global coordinates using base_start."""
    if local_span is None or base_start is None:
        return None
    return [base_start + local_span[0], base_start + local_span[1]]
