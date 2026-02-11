# /src/general.py
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
      - float NaN -> None
      - lists/tuples/objects are returned as-is (no pd.isna on list!)
    """
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def ensure_span_list(x: Any) -> Optional[List[int]]:
    """
    Enforce Python list span format: [start, end] (ints) or None.

    Accepts:
      - None / NaN -> None
      - list/tuple len 2 -> [int, int]
      - numpy array / other sequence-likes len 2 -> [int, int]
    """
    x = safe_cell(x)
    if x is None:
        return None

    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return [int(x[0]), int(x[1])]
        raise ValueError(f"Invalid span (expected length 2): {x!r}")

    try:
        xs = list(x)
        if len(xs) == 2:
            return [int(xs[0]), int(xs[1])]
    except Exception:
        pass

    raise ValueError(f"Invalid span format (expected [start,end] or None): {x!r}")


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
