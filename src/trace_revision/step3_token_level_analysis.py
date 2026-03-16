from __future__ import annotations

from typing import List, Dict, Literal, Tuple, Optional, Any
import pandas as pd
import re

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
    Step 3 pipeline (revised output schema):

    Input: Step2 operation df with columns:
      id, operation,
      source_label, source_chunk, source_index,
      target_label, target_chunk, target_index

    Output columns (fixed order):
      "operation_id", "operation", "chunk_label",
      "source_chunk", "source_chunk_index", "source", "source_index",
      "target_chunk", "target_chunk_index", "target", "target_index"
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

    def pick_chunk_label(src_lab: Optional[str], tgt_lab: Optional[str]) -> Optional[str]:
        """
        Unify labels into one field:
          - Prefer src_lab if present; else use tgt_lab.
          - (For C labels they should be identical; if not, we still pick src_lab.)
        """
        if src_lab is not None:
            return src_lab
        return tgt_lab

    def emit(
        *,
        operation: str,
        chunk_label: Optional[str],
        # chunk-level context (from Step2)
        source_chunk: Optional[str],
        source_chunk_index: Optional[List[int]],
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
                "chunk_label": chunk_label,
                "source_chunk": source_chunk,
                "source_chunk_index": source_chunk_index,
                "source": source,
                "source_index": source_index,
                "target_chunk": target_chunk,
                "target_chunk_index": target_chunk_index,
                "target": target,
                "target_index": target_index,
            }
        )
        op_id += 1

    for _, r in step2_df.iterrows():
        op = str(r["operation"])

        # ---- Step2 chunk-level fields ----
        src_lab = safe_cell(r["source_label"])
        tgt_lab = safe_cell(r["target_label"])
        chunk_label = pick_chunk_label(src_lab, tgt_lab)

        s_chunk = safe_cell(r["source_chunk"])
        t_chunk = safe_cell(r["target_chunk"])
        s_chunk_index = ensure_span_list(safe_cell(r["source_index"]))
        t_chunk_index = ensure_span_list(safe_cell(r["target_index"]))

        # Bases for local->global shifting
        s_base = span_base(s_chunk_index)
        t_base = span_base(t_chunk_index)

        if op == "Delete":
            emit(
                operation="delete",
                chunk_label=chunk_label,  # typically D*
                source_chunk=s_chunk,
                source_chunk_index=s_chunk_index,
                target_chunk=t_chunk,                 # usually None, but safe to carry if present
                target_chunk_index=t_chunk_index,     # usually None
                source=s_chunk,
                source_index=s_chunk_index,
                target=None,
                target_index=None,
            )
            continue

        if op == "Insert":
            emit(
                operation="insert",
                chunk_label=chunk_label,  # typically I*
                source_chunk=s_chunk,                 # usually None
                source_chunk_index=s_chunk_index,     # usually None
                target_chunk=t_chunk,
                target_chunk_index=t_chunk_index,
                source=None,
                source_index=None,
                target=t_chunk,
                target_index=t_chunk_index,
            )
            continue

        if op == "Move":
            # Layer 1: normalized (pure) move macro row
            norm_t_index = _normalized_move_target_span(s_chunk, t_chunk_index)
            emit(
                operation="move",
                chunk_label=chunk_label,  # typically C*
                source_chunk=s_chunk,
                source_chunk_index=s_chunk_index,
                target_chunk=t_chunk,
                target_chunk_index=t_chunk_index,
                source=s_chunk,
                source_index=s_chunk_index,
                target=s_chunk,            # pure relocation: target text equals source chunk
                target_index=norm_t_index, # normalized span
            )

            # Layer 2: token-level edits between source_chunk and OBSERVED target_chunk
            diffs = diff_with_local_spans(s_chunk or "", t_chunk or "")
            has_diff = any(d["tag"] != "preserve" for d in diffs)

            if has_diff:
                for d in diffs:
                    emit(
                        operation=d["tag"],  # preserve/insert/delete/replace
                        chunk_label=chunk_label,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
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
                    chunk_label=chunk_label,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
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
                        chunk_label=chunk_label,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
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
                    chunk_label=chunk_label,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
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
                    chunk_label=chunk_label,
                    source_chunk=s_chunk,
                    source_chunk_index=s_chunk_index,
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
                        chunk_label=chunk_label,
                        source_chunk=s_chunk,
                        source_chunk_index=s_chunk_index,
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
            "operation_id", "operation", "chunk_label",
            "source_chunk", "source_chunk_index", "source", "source_index",
            "target_chunk", "target_chunk_index", "target", "target_index",
        ],
    )





# ----------------------------
# Merge Adjacent Chunks in the Output 
# ----------------------------

_LABEL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _parse_label(lbl: Any) -> Optional[Tuple[str, int]]:
    """
    Parse labels like 'C7', 'D12', 'I3', 'R2' -> ('C', 7).
    Returns None if not parseable.
    """
    if not isinstance(lbl, str):
        return None
    m = _LABEL_RE.match(lbl.strip())
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def _span(x: Any) -> Optional[List[int]]:
    return ensure_span_list(safe_cell(x))


def _span_tuple(x: Any) -> Optional[Tuple[int, int]]:
    s = _span(x)
    if s is None:
        return None
    return (int(s[0]), int(s[1]))


def _cover_span(a: Optional[List[int]], b: Optional[List[int]]) -> Optional[List[int]]:
    if a is None:
        return b
    if b is None:
        return a
    return [min(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))]


def _concat_text(parts: List[Optional[str]]) -> str:
    # preserve original spacing/newlines as much as possible by direct concatenation
    return "".join([p for p in parts if p is not None])


# ----------------------------
# Main refinement
# ----------------------------

def refine_step3_merge_adjacent_chunks_by_label(
    step3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple chunk-label-based refinement:

    PRESERVE merging:
      - Merge adjacent preserve rows if:
          (1) operation == 'preserve'
          (2) chunk_label are consecutive (e.g., C7, C8, C9)
          (3) each chunk_label appears EXACTLY ONCE in the entire step3_df
              (meaning: no internal token-level edits were emitted for that chunk)

      - merged chunk_label becomes a Python list of labels: ['C7','C8','C9'].

    MOVE merging:
      - Identify MOVE macro rows (operation == 'move') and consider their order ignoring
        internal edit rows.
      - Merge adjacent move macros if their chunk_label are consecutive (e.g., C7, C8).
      - Emit ONE combined move macro row, then append internal edit rows for each
        member label in order (C7 edits then C8 edits, ...).
      - merged chunk_label becomes list of labels.

    Notes:
      - This function does not need Step1 sentence spans.
      - It assumes the Step3 schema includes at least:
          operation_id, operation, chunk_label,
          source_chunk, source_chunk_index, source, source_index,
          target_chunk, target_chunk_index, target, target_index
    """
    if step3_df is None or step3_df.empty:
        return step3_df

    df = step3_df.copy().reset_index(drop=True)

    required = {
        "operation", "chunk_label",
        "source_chunk", "source_chunk_index", "source", "source_index",
        "target_chunk", "target_chunk_index", "target", "target_index",
        "operation_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step3 df missing required columns for refinement: {sorted(missing)}")

    # Count how many rows each chunk_label has (used to detect "no internal edits")
    label_counts: Dict[str, int] = {}
    for lbl in df["chunk_label"].tolist():
        if isinstance(lbl, list):
            # already merged from prior runs; treat as non-mergeable further
            continue
        if isinstance(lbl, str):
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Build mapping from move macro row index -> its internal edit rows
    def collect_move_edits(macro_idx: int) -> Tuple[List[Dict[str, Any]], int]:
        macro = df.iloc[macro_idx].to_dict()
        lbl = macro.get("chunk_label")
        sci = _span_tuple(macro.get("source_chunk_index"))
        tci = _span_tuple(macro.get("target_chunk_index"))

        edits: List[Dict[str, Any]] = []
        j = macro_idx + 1
        while j < len(df):
            rj = df.iloc[j].to_dict()
            if str(rj.get("operation")) == "move":
                break
            # belong to this move chunk if label + chunk spans match
            if rj.get("chunk_label") != lbl:
                break
            if _span_tuple(rj.get("source_chunk_index")) != sci:
                break
            if _span_tuple(rj.get("target_chunk_index")) != tci:
                break
            edits.append(rj)
            j += 1
        return edits, j

    # Pass 1: merge MOVE macro rows (ignore internal edit rows for adjacency)
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(df):
        r = df.iloc[i].to_dict()
        op = str(r.get("operation"))

        if op != "move":
            out.append(r)
            i += 1
            continue

        lbl = r.get("chunk_label")
        parsed = _parse_label(lbl)
        if parsed is None:
            # unparseable label: keep as-is
            out.append(r)
            edits, nxt = collect_move_edits(i)
            out.extend(edits)
            i = nxt
            continue

        prefix, num = parsed

        # Start a run of consecutive move macros by label number
        run_macros: List[Dict[str, Any]] = [r]
        run_labels: List[str] = [lbl]
        run_edits: List[List[Dict[str, Any]]] = []

        e0, nxt0 = collect_move_edits(i)
        run_edits.append(e0)

        j = nxt0
        expected = num + 1

        while j < len(df) and str(df.iloc[j].get("operation")) == "move":
            r2 = df.iloc[j].to_dict()
            lbl2 = r2.get("chunk_label")
            p2 = _parse_label(lbl2)
            if p2 is None:
                break
            prefix2, num2 = p2
            if prefix2 != prefix or num2 != expected:
                break

            run_macros.append(r2)
            run_labels.append(lbl2)
            e2, nxt2 = collect_move_edits(j)
            run_edits.append(e2)

            expected += 1
            j = nxt2

        if len(run_macros) == 1:
            out.append(run_macros[0])
            out.extend(run_edits[0])
            i = j
            continue

        # Build combined move macro row
        first = run_macros[0]
        combined = dict(first)

        # merge spans (cover)
        src_cov = None
        tgt_cov = None
        for m in run_macros:
            src_cov = _cover_span(src_cov, _span(m.get("source_chunk_index")))
            tgt_cov = _cover_span(tgt_cov, _span(m.get("target_chunk_index")))

        combined_source_chunk = _concat_text([m.get("source_chunk") for m in run_macros])
        combined_target_chunk = _concat_text([m.get("target_chunk") for m in run_macros])

        # Pure relocation move macro: target text equals source text
        combined_source = combined_source_chunk
        combined_target = combined_source_chunk

        # Normalized target_index: anchored at first macro's target_index start
        t0 = _span(first.get("target_index"))
        if t0 is None:
            norm_t = None
        else:
            start = int(t0[0])
            norm_t = [start, start + max(len(combined_source), 1) - 1]

        combined.update({
            "operation": "move",
            "chunk_label": list(run_labels),

            "source_chunk": combined_source_chunk,
            "source_chunk_index": src_cov,
            "source": combined_source,
            "source_index": src_cov,

            "target_chunk": combined_target_chunk,
            "target_chunk_index": tgt_cov,
            "target": combined_target,
            "target_index": norm_t,
        })

        out.append(combined)

        # Append internal edits in label order: Ck edits, then Ck+1 edits, ...
        for edits in run_edits:
            out.extend(edits)

        i = j

    merged_move_df = pd.DataFrame(out, columns=df.columns).reset_index(drop=True)

    # Pass 2: merge PRESERVE rows by label adjacency + no internal edits (label appears once)
    out2: List[Dict[str, Any]] = []
    k = 0
    while k < len(merged_move_df):
        r = merged_move_df.iloc[k].to_dict()
        if str(r.get("operation")) != "preserve":
            out2.append(r)
            k += 1
            continue

        lbl = r.get("chunk_label")
        # only merge scalar labels, not already merged lists
        if not isinstance(lbl, str):
            out2.append(r)
            k += 1
            continue

        # must be "no internal edits": label appears exactly once in ORIGINAL df
        if label_counts.get(lbl, 0) != 1:
            out2.append(r)
            k += 1
            continue

        p = _parse_label(lbl)
        if p is None:
            out2.append(r)
            k += 1
            continue

        prefix, num = p

        # build run of consecutive preserve rows with no internal edits
        run_rows = [r]
        run_labels = [lbl]
        expected = num + 1

        k2 = k + 1
        while k2 < len(merged_move_df):
            r2 = merged_move_df.iloc[k2].to_dict()
            if str(r2.get("operation")) != "preserve":
                break
            lbl2 = r2.get("chunk_label")
            if not isinstance(lbl2, str):
                break
            if label_counts.get(lbl2, 0) != 1:
                break
            p2 = _parse_label(lbl2)
            if p2 is None:
                break
            prefix2, num2 = p2
            if prefix2 != prefix or num2 != expected:
                break

            run_rows.append(r2)
            run_labels.append(lbl2)
            expected += 1
            k2 += 1

        if len(run_rows) == 1:
            out2.append(run_rows[0])
            k += 1
            continue

        first = run_rows[0]
        combined = dict(first)

        src_cov = None
        tgt_cov = None
        for rr in run_rows:
            src_cov = _cover_span(src_cov, _span(rr.get("source_chunk_index")))
            tgt_cov = _cover_span(tgt_cov, _span(rr.get("target_chunk_index")))

        combined_source_chunk = _concat_text([rr.get("source_chunk") for rr in run_rows])
        combined_target_chunk = _concat_text([rr.get("target_chunk") for rr in run_rows])

        combined.update({
            "operation": "preserve",
            "chunk_label": list(run_labels),

            "source_chunk": combined_source_chunk,
            "source_chunk_index": src_cov,
            "source": combined_source_chunk,
            "source_index": src_cov,

            "target_chunk": combined_target_chunk,
            "target_chunk_index": tgt_cov,
            "target": combined_target_chunk,
            "target_index": tgt_cov,
        })

        out2.append(combined)
        k = k2

    final_df = pd.DataFrame(out2, columns=df.columns).reset_index(drop=True)

    # Resequence operation_id
    final_df["operation_id"] = list(range(1, len(final_df) + 1))
    return final_df
