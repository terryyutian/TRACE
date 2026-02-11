from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import spacy

from .general import (
    normalized_edit_similarity_from_tokens,
    ensure_span_list,
)

# ============================================================
# Core data structures
# ============================================================

@dataclass(frozen=True)
class UnitSpan:
    """Inclusive span over unit indices: [start, end]."""
    start: int
    end: int

    def overlaps(self, other: "UnitSpan") -> bool:
        return not (self.end < other.start or other.end < self.start)


@dataclass(frozen=True)
class TextUnit:
    """A sentence unit (after optional short-sentence merging)."""
    text: str
    tokens: List[str]


@dataclass
class Match:
    """Final accepted match between a source span and a target span."""
    src_span: UnitSpan
    tgt_span: UnitSpan
    score: float
    kind: str  # "1-1", "split", "merge"


def build_spacy_nlp(model: str = "en_core_web_sm"):
    return spacy.load(model)


# ============================================================
# Sentence segmentation + short-sentence merge (spaCy)
# ============================================================

def sentence_tokenize_with_short_merge(
    text: str,
    nlp,
    min_tokens: int = 5,
    lowercase: bool = True,
    drop_space_tokens: bool = True,
) -> List[TextUnit]:
    """
    spaCy sentence segmentation + short-sentence merge:
      - If a sentence has < min_tokens tokens, merge forward; if last, merge backward.
    """
    doc = nlp(text)
    raw_sents = [sent for sent in doc.sents if sent.text.strip()]
    if not raw_sents:
        return []

    def sent_to_unit(sent) -> TextUnit:
        toks: List[str] = []
        for tok in sent:
            if drop_space_tokens and tok.is_space:
                continue
            t = tok.text.lower() if lowercase else tok.text
            toks.append(t)
        return TextUnit(text=sent.text.strip(), tokens=toks)

    units = [sent_to_unit(s) for s in raw_sents]

    merged: List[TextUnit] = []
    i = 0
    while i < len(units):
        u = units[i]
        if len(u.tokens) < min_tokens:
            if i < len(units) - 1:
                text_acc = u.text
                tok_acc = list(u.tokens)
                i += 1
                while i < len(units) and len(tok_acc) < min_tokens:
                    text_acc = (text_acc + " " + units[i].text).strip()
                    tok_acc.extend(units[i].tokens)
                    i += 1
                merged.append(TextUnit(text=text_acc, tokens=tok_acc))
            else:
                if merged:
                    prev = merged.pop()
                    merged.append(TextUnit(
                        text=(prev.text + " " + u.text).strip(),
                        tokens=prev.tokens + u.tokens
                    ))
                else:
                    merged.append(u)
                i += 1
        else:
            merged.append(u)
            i += 1

    return merged


# ============================================================
# Split/merge refinement helpers
# ============================================================

def _concat_units(units: List[TextUnit], span: UnitSpan) -> TextUnit:
    texts = [u.text for u in units[span.start:span.end + 1]]
    toks: List[str] = []
    for u in units[span.start:span.end + 1]:
        toks.extend(u.tokens)
    return TextUnit(text=" ".join(texts).strip(), tokens=toks)


def _expand_window_by_improvement(
    anchor_tokens: List[str],
    units: List[TextUnit],
    center: int,
    direction: int,
    strict_increase: bool = True,
) -> int:
    """
    Expand from 'center' in 'direction' (+1 forward, -1 backward) while similarity improves.
    Returns number of steps expanded (>=0).
    """
    base_tokens = units[center].tokens
    best = normalized_edit_similarity_from_tokens(anchor_tokens, base_tokens)

    steps = 0
    running_tokens = list(base_tokens)

    while True:
        nxt = center + (steps + 1) * direction
        if nxt < 0 or nxt >= len(units):
            break

        trial_tokens = (
            running_tokens + units[nxt].tokens
            if direction == 1
            else units[nxt].tokens + running_tokens
        )
        sc = normalized_edit_similarity_from_tokens(anchor_tokens, trial_tokens)
        improved = sc > best if strict_increase else sc >= best
        if not improved:
            break

        best = sc
        running_tokens = trial_tokens
        steps += 1

    return steps


def _best_span_within_window(
    anchor_tokens: List[str],
    units: List[TextUnit],
    L: int,
    R: int,
) -> Tuple[UnitSpan, float]:
    """Search all contiguous spans within [L,R] and return the max-similarity span."""
    best_span = UnitSpan(L, L)
    best_score = -1.0

    for a in range(L, R + 1):
        running_tokens: List[str] = []
        for b in range(a, R + 1):
            running_tokens.extend(units[b].tokens)
            sc = normalized_edit_similarity_from_tokens(anchor_tokens, running_tokens)
            if sc > best_score:
                best_score = sc
                best_span = UnitSpan(a, b)

    return best_span, best_score


# ============================================================
# Matching (Step 1)
# ============================================================

def _initial_best_matches(
    src_units: List[TextUnit],
    tgt_units: List[TextUnit],
) -> List[Tuple[int, int, float]]:
    """For each source unit i, choose target j maximizing token similarity."""
    if not tgt_units:
        return [(i, -1, -1.0) for i in range(len(src_units))]

    results: List[Tuple[int, int, float]] = []
    for i, su in enumerate(src_units):
        best_j = 0
        best_sc = -1.0
        for j, tu in enumerate(tgt_units):
            sc = normalized_edit_similarity_from_tokens(su.tokens, tu.tokens)
            if sc > best_sc:
                best_sc = sc
                best_j = j
        results.append((i, best_j, best_sc))
    return results


def _build_candidates(
    src_units: List[TextUnit],
    tgt_units: List[TextUnit],
    theta: float,
    delta_len: int,
    strict_increase: bool,
) -> List[Match]:
    """
    Conservative design:
      1) initial best Tj for each Si
      2) if |Si| >= |Tj| + delta => split refinement (target span)
      3) if |Tj| >= |Si| + delta => merge refinement (source span)
      4) keep only if score >= theta
    """
    init = _initial_best_matches(src_units, tgt_units)
    candidates: List[Match] = []

    for i, j, base_sc in init:
        if j < 0:
            continue

        su = src_units[i]
        tu = tgt_units[j]
        li, lj = len(su.tokens), len(tu.tokens)

        best = Match(UnitSpan(i, i), UnitSpan(j, j), base_sc, kind="1-1")

        if (li - lj) >= delta_len:
            f = _expand_window_by_improvement(su.tokens, tgt_units, j, direction=1, strict_increase=strict_increase)
            b = _expand_window_by_improvement(su.tokens, tgt_units, j, direction=-1, strict_increase=strict_increase)
            L, R = max(0, j - b), min(len(tgt_units) - 1, j + f)
            span, sc = _best_span_within_window(su.tokens, tgt_units, L, R)
            if sc > best.score:
                best = Match(UnitSpan(i, i), span, sc, kind="split")

        elif (lj - li) >= delta_len:
            f = _expand_window_by_improvement(tu.tokens, src_units, i, direction=1, strict_increase=strict_increase)
            b = _expand_window_by_improvement(tu.tokens, src_units, i, direction=-1, strict_increase=strict_increase)
            L, R = max(0, i - b), min(len(src_units) - 1, i + f)
            span, sc = _best_span_within_window(tu.tokens, src_units, L, R)
            if sc > best.score:
                best = Match(span, UnitSpan(j, j), sc, kind="merge")

        if best.score >= theta:
            candidates.append(best)

    return candidates


def _resolve_conflicts_greedy_nonoverlap(candidates: List[Match]) -> List[Match]:
    """Greedy: sort by score desc; accept if no overlap in src spans and no overlap in tgt spans."""
    sorted_c = sorted(candidates, key=lambda m: m.score, reverse=True)
    accepted: List[Match] = []
    used_src: List[UnitSpan] = []
    used_tgt: List[UnitSpan] = []

    for m in sorted_c:
        if any(m.src_span.overlaps(s) for s in used_src):
            continue
        if any(m.tgt_span.overlaps(t) for t in used_tgt):
            continue
        accepted.append(m)
        used_src.append(m.src_span)
        used_tgt.append(m.tgt_span)

    accepted.sort(key=lambda x: (x.src_span.start, x.tgt_span.start))
    return accepted


def _char_spans_for_units(units: List[TextUnit]) -> List[Tuple[int, int]]:
    """Character spans per unit in reconstructed text: ' '.join(unit.text). Inclusive [start,end]."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for idx, u in enumerate(units):
        start = pos
        end = start + len(u.text) - 1
        spans.append((start, end))
        pos = end + 1
        if idx != len(units) - 1:
            pos += 1  # space between units
    return spans


def _span_to_char_index(span: UnitSpan, unit_char_spans: List[Tuple[int, int]]) -> List[int]:
    return [unit_char_spans[span.start][0], unit_char_spans[span.end][1]]


def _build_unmatched_spans(total: int, covered: set, group: bool) -> List[UnitSpan]:
    spans: List[UnitSpan] = []
    i = 0
    while i < total:
        if i in covered:
            i += 1
            continue
        j = i
        while j + 1 < total and (j + 1) not in covered:
            j += 1
        if group:
            spans.append(UnitSpan(i, j))
        else:
            for k in range(i, j + 1):
                spans.append(UnitSpan(k, k))
        i = j + 1
    return spans


# ============================================================
# Post-hoc grouping adjacent unmatched rows (Option 1)
# ============================================================

def group_adjacent_unmatched(df: pd.DataFrame, prefix: str = "D") -> pd.DataFrame:
    """
    Collapse maximal consecutive runs whose label starts with `prefix` (e.g., D*, I*).
    Keeps FIRST label in run, concatenates chunk text, and merges chunk_index.
    """
    required = {"chunk", "label", "chunk_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    i = 0
    n = len(df)

    def is_prefix(lbl: Any) -> bool:
        return isinstance(lbl, str) and lbl.startswith(prefix)

    while i < n:
        row = df.iloc[i]
        lbl = row["label"]

        if not is_prefix(lbl):
            rows.append({
                "chunk": row["chunk"],
                "label": row["label"],
                "chunk_index": ensure_span_list(row["chunk_index"]),
            })
            i += 1
            continue

        run_label = row["label"]
        idx0 = ensure_span_list(row["chunk_index"])
        if idx0 is None:
            raise ValueError(f"Invalid chunk_index at row {i}: {row['chunk_index']!r}")
        start_char, end_char = idx0[0], idx0[1]

        chunk_parts = [str(row["chunk"])]
        i += 1

        while i < n and is_prefix(df.iloc[i]["label"]):
            r = df.iloc[i]
            chunk_parts.append(str(r["chunk"]))
            idx = ensure_span_list(r["chunk_index"])
            if idx is None:
                raise ValueError(f"Invalid chunk_index at row {i}: {r['chunk_index']!r}")
            end_char = idx[1]
            i += 1

        rows.append({
            "chunk": " ".join([p for p in chunk_parts if p.strip()]),
            "label": run_label,
            "chunk_index": [int(start_char), int(end_char)],
        })

    return pd.DataFrame(rows, columns=["chunk", "label", "chunk_index"]).reset_index(drop=True)


# ============================================================
# Step 1 main (public)
# ============================================================

def step1_chunk_matching_to_dataframes(
    src_text: str,
    tgt_text: str,
    *,
    nlp=None,
    min_sent_tokens: int = 5,
    theta: float = 0.6,
    delta_len: int = 5,
    strict_increase: bool = True,
    group_unmatched: bool = False,
    group_adjacent_unmatched_posthoc: bool = True,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Step 1 pipeline output:
      - src_df: columns [chunk, label, chunk_index] with C*/D* labels
      - tgt_df: columns [chunk, label, chunk_index] with C*/I* labels

    chunk_index is ALWAYS a Python list [start,end] (inclusive) after this function returns.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    src_units = sentence_tokenize_with_short_merge(src_text, nlp, min_tokens=min_sent_tokens)
    tgt_units = sentence_tokenize_with_short_merge(tgt_text, nlp, min_tokens=min_sent_tokens)

    candidates = _build_candidates(
        src_units, tgt_units,
        theta=theta,
        delta_len=delta_len,
        strict_increase=strict_increase,
    )
    matches = _resolve_conflicts_greedy_nonoverlap(candidates)

    # Assign C labels based on source order of matched chunks
    matches_src_sorted = sorted(matches, key=lambda m: (m.src_span.start, m.src_span.end))
    c_label_for_match: Dict[tuple, str] = {}
    for k, m in enumerate(matches_src_sorted, start=1):
        key = (m.src_span.start, m.src_span.end, m.tgt_span.start, m.tgt_span.end, m.score, m.kind)
        c_label_for_match[key] = f"C{k}"

    covered_src = set()
    covered_tgt = set()
    src_match_by_start: Dict[int, Match] = {}
    tgt_match_by_start: Dict[int, Match] = {}
    for m in matches:
        covered_src.update(range(m.src_span.start, m.src_span.end + 1))
        covered_tgt.update(range(m.tgt_span.start, m.tgt_span.end + 1))
        src_match_by_start[m.src_span.start] = m
        tgt_match_by_start[m.tgt_span.start] = m

    src_unit_spans = _char_spans_for_units(src_units)
    tgt_unit_spans = _char_spans_for_units(tgt_units)

    src_unmatched_spans = _build_unmatched_spans(len(src_units), covered_src, group_unmatched)
    tgt_unmatched_spans = _build_unmatched_spans(len(tgt_units), covered_tgt, group_unmatched)

    src_unmatched_by_start = {sp.start: sp for sp in src_unmatched_spans}
    tgt_unmatched_by_start = {sp.start: sp for sp in tgt_unmatched_spans}

    def c_label(m: Match) -> str:
        key = (m.src_span.start, m.src_span.end, m.tgt_span.start, m.tgt_span.end, m.score, m.kind)
        if key in c_label_for_match:
            return c_label_for_match[key]
        # fallback if float keys differ slightly
        for mm, lbl in c_label_for_match.items():
            if mm[0] == m.src_span.start and mm[1] == m.src_span.end and mm[2] == m.tgt_span.start and mm[3] == m.tgt_span.end:
                return lbl
        return "C?"

    # Build src_df
    src_rows: List[Dict[str, Any]] = []
    d_count = 0
    i = 0
    while i < len(src_units):
        if i in covered_src and i in src_match_by_start:
            m = src_match_by_start[i]
            lbl = c_label(m)
            chunk_unit = _concat_units(src_units, m.src_span)
            src_rows.append({
                "chunk": chunk_unit.text,
                "label": lbl,
                "chunk_index": _span_to_char_index(m.src_span, src_unit_spans)
            })
            i = m.src_span.end + 1
        elif i not in covered_src:
            sp = src_unmatched_by_start[i]
            d_count += 1
            chunk_unit = _concat_units(src_units, sp)
            src_rows.append({
                "chunk": chunk_unit.text,
                "label": f"D{d_count}",
                "chunk_index": _span_to_char_index(sp, src_unit_spans)
            })
            i = sp.end + 1
        else:
            i += 1

    src_df = pd.DataFrame(src_rows, columns=["chunk", "label", "chunk_index"])

    # Build tgt_df
    tgt_rows: List[Dict[str, Any]] = []
    ins_count = 0
    j = 0
    while j < len(tgt_units):
        if j in covered_tgt and j in tgt_match_by_start:
            m = tgt_match_by_start[j]
            lbl = c_label(m)
            chunk_unit = _concat_units(tgt_units, m.tgt_span)
            tgt_rows.append({
                "chunk": chunk_unit.text,
                "label": lbl,
                "chunk_index": _span_to_char_index(m.tgt_span, tgt_unit_spans)
            })
            j = m.tgt_span.end + 1
        elif j not in covered_tgt:
            sp = tgt_unmatched_by_start[j]
            ins_count += 1
            chunk_unit = _concat_units(tgt_units, sp)
            tgt_rows.append({
                "chunk": chunk_unit.text,
                "label": f"I{ins_count}",
                "chunk_index": _span_to_char_index(sp, tgt_unit_spans)
            })
            j = sp.end + 1
        else:
            j += 1

    tgt_df = pd.DataFrame(tgt_rows, columns=["chunk", "label", "chunk_index"])

    # Post-hoc grouping adjacent unmatched rows (Option 1)
    if group_adjacent_unmatched_posthoc:
        src_df = group_adjacent_unmatched(src_df, prefix="D")
        tgt_df = group_adjacent_unmatched(tgt_df, prefix="I")

    # Enforce span list format at the end (defensive)
    src_df["chunk_index"] = src_df["chunk_index"].apply(ensure_span_list)
    tgt_df["chunk_index"] = tgt_df["chunk_index"].apply(ensure_span_list)

    debug_info = None
    if debug:
        debug_info = {
            "src_units": [u.text for u in src_units],
            "tgt_units": [u.text for u in tgt_units],
            "candidates": candidates,
            "matches": matches,
            "params": {
                "min_sent_tokens": min_sent_tokens,
                "theta": theta,
                "delta_len": delta_len,
                "strict_increase": strict_increase,
                "group_unmatched": group_unmatched,
                "group_adjacent_unmatched_posthoc": group_adjacent_unmatched_posthoc,
            }
        }

    return src_df, tgt_df, debug_info
