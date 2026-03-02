from __future__ import annotations

from typing import List, Dict, Literal, Tuple, Optional, Any
import pandas as pd

from .general import (
    Tok,
    tokenize_with_spans,
    normalized_edit_similarity,
    ensure_span_list,
    safe_cell,
    span_base,
    add_span_base,
)

Tag = Literal["preserve", "insert", "delete", "replace"]


# ============================================================
# Token-level Levenshtein alignment returning token indices
# ============================================================

def _levenshtein_token_ops_indexed(
    a: List[str], b: List[str]
) -> List[Tuple[Tag, Optional[int], Optional[int]]]:
    """
    Token-level Levenshtein alignment returning atomic ops:
      (tag, a_index_or_None, b_index_or_None)

    Tie-break policy (stable):
      1) prefer diagonal (preserve/replace)
      2) then delete
      3) then insert
    """
    m, n = len(a), len(b)

    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        D[i][0] = i
    for j in range(1, n + 1):
        D[0][j] = j

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            D[i][j] = min(
                D[i - 1][j] + 1,        # delete
                D[i][j - 1] + 1,        # insert
                D[i - 1][j - 1] + cost  # preserve/replace
            )

    ops: List[Tuple[Tag, Optional[int], Optional[int]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        # diagonal first
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + cost:
                tag: Tag = "preserve" if cost == 0 else "replace"
                ops.append((tag, i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        # delete
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("delete", i - 1, None))
            i -= 1
            continue

        # insert
        if j > 0 and D[i][j] == D[i][j - 1] + 1:
            ops.append(("insert", None, j - 1))
            j -= 1
            continue

        raise RuntimeError("Levenshtein backtrace failed.")

    ops.reverse()
    return ops


# ============================================================
# Merge adjacent ops + compute LOCAL char spans, and slice text
# ============================================================

def _merge_adjacent_ops_indexed(
    ops: List[Tuple[Tag, Optional[int], Optional[int]]],
    a_toks: List[Tok],
    b_toks: List[Tok],
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    Merge adjacent ops with same tag into span rows AND compute local char spans.
    Returns dict rows:
      tag, source_text, target_text, source_span, target_span
    spans are [start,end] inclusive within the LOCAL strings, or None.
    source_text/target_text are exact slices from the input strings.
    """
    rows: List[Dict[str, Any]] = []

    cur_tag: Optional[Tag] = None
    cur_a_idx: List[int] = []
    cur_b_idx: List[int] = []

    def flush():
        nonlocal cur_tag, cur_a_idx, cur_b_idx
        if cur_tag is None:
            return

        src_span: Optional[List[int]] = None
        if cur_a_idx:
            s = a_toks[cur_a_idx[0]].start
            e = a_toks[cur_a_idx[-1]].end
            src_span = [s, e]

        tgt_span: Optional[List[int]] = None
        if cur_b_idx:
            s = b_toks[cur_b_idx[0]].start
            e = b_toks[cur_b_idx[-1]].end
            tgt_span = [s, e]

        src_text = source[src_span[0] : src_span[1] + 1] if src_span else ""
        tgt_text = target[tgt_span[0] : tgt_span[1] + 1] if tgt_span else ""

        rows.append(
            {
                "tag": cur_tag,
                "source_text": src_text,
                "target_text": tgt_text,
                "source_span": src_span,
                "target_span": tgt_span,
            }
        )

        cur_tag = None
        cur_a_idx = []
        cur_b_idx = []

    for tag, ai, bj in ops:
        if cur_tag is None:
            cur_tag = tag
            if ai is not None:
                cur_a_idx.append(ai)
            if bj is not None:
                cur_b_idx.append(bj)
            continue

        if tag == cur_tag:
            if ai is not None:
                cur_a_idx.append(ai)
            if bj is not None:
                cur_b_idx.append(bj)
        else:
            flush()
            cur_tag = tag
            if ai is not None:
                cur_a_idx.append(ai)
            if bj is not None:
                cur_b_idx.append(bj)

    flush()
    return [r for r in rows if not (r["source_text"] == "" and r["target_text"] == "")]


def diff_with_local_spans(source: str, target: str) -> List[Dict[str, Any]]:
    """Diff source->target and return merged rows with LOCAL spans + exact sliced text."""
    a_toks = tokenize_with_spans(source or "")
    b_toks = tokenize_with_spans(target or "")
    a = [t.text for t in a_toks]
    b = [t.text for t in b_toks]
    ops = _levenshtein_token_ops_indexed(a, b)
    return _merge_adjacent_ops_indexed(ops, a_toks, b_toks, source or "", target or "")


# ============================================================
# Helpers for updated Move decomposition
# ============================================================

def _normalized_move_target_span(
    source_chunk: Optional[str],
    target_index: Optional[List[int]],
) -> Optional[List[int]]:
    """
    For the macro 'move' row, represent a pure relocation:
      - anchor at target_index.start
      - end = start + len(source_chunk) - 1   (inclusive)
    If indices/text are missing, fall back to the Step2 target_index as-is.
    """
    if not target_index or not isinstance(target_index, list) or len(target_index) != 2:
        return target_index

    if source_chunk is None:
        return target_index

    start = int(target_index[0])
    # len() counts characters; Step2 indices are inclusive.
    length = len(source_chunk)
    if length <= 0:
        return target_index

    return [start, start + length - 1]


# ============================================================
# Step 3 main (public)
# ============================================================

def step3_fine_grained_revision_analysis(
    step2_df: pd.DataFrame,
    *,
    replace_similarity_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Step 3 pipeline:
      Input: Step2 operation df with:
        id, operation,
        source_label, source_chunk, source_index,
        target_label, target_chunk, target_index, ...

      Output columns (fixed order):
        "operation_id", "operation",
        "source_chunk_label", "source_chunk", "source_chunk_index",
        "source", "source_index",
        "target_chunk_label", "target_chunk", "target_chunk_index",
        "target", "target_index"
    """
    required = {
        "operation",
        "source_label", "source_chunk", "source_index",
        "target_label", "target_chunk", "target_index",
    }
    missing = required - set(step2_df.columns)
    if missing:
        raise ValueError(f"step2_df missing required columns: {sorted(missing)}")

    out_rows: List[Dict[str, Any]] = []
    op_id = 1

    def emit(
        *,
        operation: str,
        # chunk-level context (from Step2)
        source_chunk_label: Optional[str],
        source_chunk: Optional[str],
        source_chunk_index: Optional[List[int]],
        target_chunk_label: Optional[str],
        target_chunk: Optional[str],
        target_chunk_index: Optional[List[int]],
        # fine-grained fields (this row)
        source: Optional[str],
        source_index: Optional[List[int]],
        target: Optional[str],
        target_index: Optional[List[int]],
    ) -> None:
        nonlocal op_id
        out_rows.append(
            {
                "operation_id": op_id,
                "operation": operation,
                "source_chunk_label": source_chunk_label,
                "source_chunk": source_chunk,
                "source_chunk_index": source_chunk_index,
                "source": source,
                "source_index": source_index,
                "target_chunk_label": target_chunk_label,
                "target_chunk": target_chunk,
                "target_chunk_index": target_chunk_index,
                "target": target,
                "target_index": target_index,
            }
        )
        op_id += 1

    for _, r in step2_df.iterrows():
        op = str(r["operation"])

        # ---- chunk-level context from Step2 (copied to every Step3 row) ----
        s_lab = safe_cell(r["source_label"])
        t_lab = safe_cell(r["target_label"])

        s_chunk = safe_cell(r["source_chunk"])
        t_chunk = safe_cell(r["target_chunk"])

        s_chunk_index = ensure_span_list(safe_cell(r["source_index"]))
        t_chunk_index = ensure_span_list(safe_cell(r["target_index"]))

        # Bases used for local->global shifting of token spans
        s_base = span_base(s_chunk_index)
        t_base = span_base(t_chunk_index)

        if op == "Delete":
            emit(
                operation="delete",
                source_chunk_label=s_lab,
                source_chunk=s_chunk,
                source_chunk_index=s_chunk_index,
                target_chunk_label=None,
                target_chunk=None,
                target_chunk_index=None,
                source=s_chunk,
                source_index=s_chunk_index,
                target=None,
                target_index=None,
            )
            continue

        if op == "Insert":
            emit(
                operation="insert",
                source_chunk_label=None,
                source_chunk=None,
                source_chunk_index=None,
                target_chunk_label=t_lab,
                target_chunk=t_chunk,
                target_chunk_index=t_chunk_index,
                source=None,
                source_index=None,
                target=t_chunk,
                target_index=t_chunk_index,
            )
            continue

        if op == "Move":
            # Layer 1: normalized (pure) move macro row (Option A)
            norm_t_index = _normalized_move_target_span(s_chunk, t_chunk_index)

            emit(
                operation="move",
                source_chunk_label=s_lab,
                source_chunk=s_chunk,
                source_chunk_index=s_chunk_index,
                target_chunk_label=t_lab,
                target_chunk=t_chunk,
                target_chunk_index=t_chunk_index,
                source=s_chunk,
                source_index=s_chunk_index,
                target=s_chunk,            # exact same as source_chunk
                target_index=norm_t_index, # normalized span
            )

            # Layer 2: token-level edits between source_chunk and OBSERVED target_chunk
            diffs = diff_with_local_spans(s_chunk or "", t_chunk or "")
            has_diff = any(d["tag"] != "preserve" for d in diffs)

            if has_diff:
                for d in diffs:
                    emit(
                        operation=d["tag"],  # preserve/insert/delete/replace
                        source_chunk_label=s_lab,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
                        target_chunk_label=t_lab,
                        target_chunk=t_chunk,
                        target_chunk_index=t_chunk_index,
                        source=d["source_text"] if d["source_text"] != "" else None,
                        source_index=add_span_base(d["source_span"], s_base),
                        target=d["target_text"] if d["target_text"] != "" else None,
                        target_index=add_span_base(d["target_span"], t_base),
                    )
            continue

        if op == "Equal":
            diffs = diff_with_local_spans(s_chunk or "", t_chunk or "")
            has_diff = any(d["tag"] != "preserve" for d in diffs)

            if not has_diff:
                emit(
                    operation="preserve",
                    source_chunk_label=s_lab,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
                    target_chunk_label=t_lab,
                    target_chunk=t_chunk,
                    target_chunk_index=t_chunk_index,
                    source=s_chunk,
                    source_index=s_chunk_index,
                    target=t_chunk,
                    target_index=t_chunk_index,
                )
            else:
                for d in diffs:
                    emit(
                        operation=d["tag"],
                        source_chunk_label=s_lab,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
                        target_chunk_label=t_lab,
                        target_chunk=t_chunk,
                        target_chunk_index=t_chunk_index,
                        source=d["source_text"] if d["source_text"] != "" else None,
                        source_index=add_span_base(d["source_span"], s_base),
                        target=d["target_text"] if d["target_text"] != "" else None,
                        target_index=add_span_base(d["target_span"], t_base),
                    )
            continue

        if op == "Replace":
            sim = normalized_edit_similarity(s_chunk or "", t_chunk or "", lowercase=True)

            # If too different, represent as ONE macro replace row
            if sim < replace_similarity_threshold:
                emit(
                    operation="replace",
                    source_chunk_label=s_lab,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
                    target_chunk_label=t_lab,
                    target_chunk=t_chunk,
                    target_chunk_index=t_chunk_index,
                    source=s_chunk,
                    source_index=s_chunk_index,
                    target=t_chunk,
                    target_index=t_chunk_index,
                )
                continue

            # Otherwise expand token-level rows (no macro row)
            diffs = diff_with_local_spans(s_chunk or "", t_chunk or "")
            if not diffs:
                emit(
                    operation="preserve",
                    source_chunk_label=s_lab,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
                    target_chunk_label=t_lab,
                    target_chunk=t_chunk,
                    target_chunk_index=t_chunk_index,
                    source=s_chunk,
                    source_index=s_chunk_index,
                    target=t_chunk,
                    target_index=t_chunk_index,
                )
            else:
                for d in diffs:
                    emit(
                        operation=d["tag"],
                        source_chunk_label=s_lab,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
                        target_chunk_label=t_lab,
                        target_chunk=t_chunk,
                        target_chunk_index=t_chunk_index,
                        source=d["source_text"] if d["source_text"] != "" else None,
                        source_index=add_span_base(d["source_span"], s_base),
                        target=d["target_text"] if d["target_text"] != "" else None,
                        target_index=add_span_base(d["target_span"], t_base),
                    )
            continue

        raise ValueError(f"Unknown operation type in step2_df: {op}")

    return pd.DataFrame(
        out_rows,
        columns=[
            "operation_id", "operation",
            "source_chunk_label", "source_chunk", "source_chunk_index",
            "source", "source_index",
            "target_chunk_label", "target_chunk", "target_chunk_index",
            "target", "target_index",
        ],
    )