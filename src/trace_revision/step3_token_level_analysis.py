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


# ============================================================
# Step 3 post-hoc readability refinement
#   - merge adjacent MOVE macro rows by SOURCE sentence adjacency only
#   - optionally merge adjacent PRESERVE rows by sentence adjacency (both src+tgt)
# ============================================================


def _tspan(x: Any) -> Optional[Tuple[int, int]]:
    x = ensure_span_list(safe_cell(x))
    if x is None:
        return None
    return (int(x[0]), int(x[1]))


def refine_step3_merge_adjacent_sentence_units(
    step3_df: pd.DataFrame,
    *,
    src_sentence_spans: List[Tuple[int, int]],
    tgt_sentence_spans: List[Tuple[int, int]],
    merge_preserve: bool = True,
) -> pd.DataFrame:
    """
    Readability refinement on Step3 output.

    MOVE merging (SOURCE adjacency only):
      - Merge adjacent MOVE *macro* rows when their source_chunk_index spans are consecutive
        source sentence units. Target adjacency is NOT required.
      - Works even when each moved sentence has internal edits:
          output one combined 'move' macro row, then internal edits for each sentence
          in source sentence order.
      - Merge only when each move macro is a whole sentence on BOTH sides:
          source_chunk_index matches a source sentence span AND target_chunk_index matches
          a target sentence span.

    PRESERVE merging (optional):
      - Merge adjacent 'preserve' rows only when BOTH source_index and target_index are
        whole-sentence spans and consecutive in BOTH src and tgt.
      - Never merge partial sentences.

    Assumes Step3 schema columns:
      operation_id, operation, chunk_label,
      source_chunk, source_chunk_index, source, source_index,
      target_chunk, target_chunk_index, target, target_index
    """
    if step3_df is None or step3_df.empty:
        return step3_df

    df = step3_df.copy().reset_index(drop=True)

    needed = {
        "operation", "chunk_label",
        "source_chunk", "source_chunk_index", "source", "source_index",
        "target_chunk", "target_chunk_index", "target", "target_index",
        "operation_id",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Step3 df missing required columns for refinement: {sorted(missing)}")

    src_sid = {sp: i for i, sp in enumerate(src_sentence_spans)}
    tgt_sid = {sp: i for i, sp in enumerate(tgt_sentence_spans)}

    def is_move_macro(r: Dict[str, Any]) -> bool:
        return str(r.get("operation")) == "move"

    def is_whole_sentence_move_macro(r: Dict[str, Any]) -> bool:
        s = _tspan(r.get("source_chunk_index"))
        t = _tspan(r.get("target_chunk_index"))
        return (s in src_sid) and (t in tgt_sid)

    def collect_move_edits(macro_idx: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Collect rows immediately following a move macro that belong to that SAME sentence move.
        Identify by matching chunk_label + source_chunk_index + target_chunk_index.
        Stop at next move macro or context change.
        """
        macro = df.iloc[macro_idx].to_dict()
        cl = macro.get("chunk_label")
        sci = _tspan(macro.get("source_chunk_index"))
        tci = _tspan(macro.get("target_chunk_index"))

        edits: List[Dict[str, Any]] = []
        j = macro_idx + 1
        while j < len(df):
            rj = df.iloc[j].to_dict()
            if is_move_macro(rj):
                break
            if rj.get("chunk_label") != cl:
                break
            if _tspan(rj.get("source_chunk_index")) != sci:
                break
            if _tspan(rj.get("target_chunk_index")) != tci:
                break
            edits.append(rj)
            j += 1
        return edits, j

    out: List[Dict[str, Any]] = []
    i = 0

    # ---- Pass A: merge MOVE macro runs (adjacent in SOURCE only) ----
    while i < len(df):
        r = df.iloc[i].to_dict()
        if not is_move_macro(r):
            out.append(r)
            i += 1
            continue

        # Not a whole sentence move -> keep as-is (+ its edits)
        if not is_whole_sentence_move_macro(r):
            out.append(r)
            edits, nxt = collect_move_edits(i)
            out.extend(edits)
            i = nxt
            continue

        # Start a run of source-adjacent sentence move macros
        run_macros = [r]
        run_edits: List[List[Dict[str, Any]]] = []

        e0, nxt0 = collect_move_edits(i)
        run_edits.append(e0)

        prev_src = src_sid[_tspan(r["source_chunk_index"])]

        j = nxt0
        while j < len(df) and is_move_macro(df.iloc[j].to_dict()):
            r2 = df.iloc[j].to_dict()
            if not is_whole_sentence_move_macro(r2):
                break
            s2 = _tspan(r2["source_chunk_index"])
            src2 = src_sid[s2]
            if src2 != prev_src + 1:
                break

            run_macros.append(r2)
            e2, nxt2 = collect_move_edits(j)
            run_edits.append(e2)

            prev_src = src2
            j = nxt2

        if len(run_macros) == 1:
            out.append(run_macros[0])
            out.extend(run_edits[0])
            i = j
            continue

        # ---- Create combined move macro row ----
        first, last = run_macros[0], run_macros[-1]

        src_start = _tspan(first["source_chunk_index"])[0]
        src_end = _tspan(last["source_chunk_index"])[1]

        # Target adjacency not required -> summarize target span as covering range
        tgt_starts = [_tspan(m["target_chunk_index"])[0] for m in run_macros]
        tgt_ends = [_tspan(m["target_chunk_index"])[1] for m in run_macros]
        tgt_start = min(tgt_starts)
        tgt_end_obs = max(tgt_ends)

        combined_source_chunk = "".join([m.get("source_chunk") or "" for m in run_macros])
        combined_target_chunk = "".join([m.get("target_chunk") or "" for m in run_macros])

        # Pure relocation macro: target text equals source text
        combined_source = combined_source_chunk
        combined_target = combined_source_chunk

        # Normalized move target span: anchored at tgt_start, length from combined_source
        norm_tgt_end = tgt_start + max(len(combined_source), 1) - 1

        labels = [m.get("chunk_label") for m in run_macros]
        combined_label = labels[0] if all(l == labels[0] for l in labels) else f"{labels[0]}-{labels[-1]}"

        combined = dict(first)
        combined.update({
            "operation": "move",
            "chunk_label": combined_label,

            "source_chunk": combined_source_chunk,
            "source_chunk_index": [src_start, src_end],
            "source": combined_source,
            "source_index": [src_start, src_end],

            "target_chunk": combined_target_chunk,
            "target_chunk_index": [tgt_start, tgt_end_obs],

            "target": combined_target,
            "target_index": [tgt_start, norm_tgt_end],
        })

        out.append(combined)

        # Append internal edits sentence-by-sentence in source order
        for edits in run_edits:
            out.extend(edits)

        i = j

    merged = pd.DataFrame(out, columns=df.columns).reset_index(drop=True)
    merged["operation_id"] = list(range(1, len(merged) + 1))

    if not merge_preserve:
        return merged

    # ---- Pass B: merge sentence-level PRESERVES (adjacent in BOTH src+tgt) ----
    out2: List[Dict[str, Any]] = []
    k = 0
    merged = merged.reset_index(drop=True)

    while k < len(merged):
        r = merged.iloc[k].to_dict()
        if str(r.get("operation")) != "preserve":
            out2.append(r)
            k += 1
            continue

        s = _tspan(r.get("source_index"))
        t = _tspan(r.get("target_index"))
        if (s not in src_sid) or (t not in tgt_sid):
            out2.append(r)
            k += 1
            continue

        cl = r.get("chunk_label")
        prev_s = src_sid[s]
        prev_t = tgt_sid[t]
        run = [r]

        k2 = k + 1
        while k2 < len(merged):
            r2 = merged.iloc[k2].to_dict()
            if str(r2.get("operation")) != "preserve":
                break
            if r2.get("chunk_label") != cl:
                break
            s2 = _tspan(r2.get("source_index"))
            t2 = _tspan(r2.get("target_index"))
            if (s2 not in src_sid) or (t2 not in tgt_sid):
                break
            if src_sid[s2] != prev_s + 1:
                break
            if tgt_sid[t2] != prev_t + 1:
                break
            run.append(r2)
            prev_s = src_sid[s2]
            prev_t = tgt_sid[t2]
            k2 += 1

        if len(run) == 1:
            out2.append(run[0])
            k += 1
            continue

        first, last = run[0], run[-1]
        s_start = _tspan(first["source_index"])[0]
        s_end = _tspan(last["source_index"])[1]
        t_start = _tspan(first["target_index"])[0]
        t_end = _tspan(last["target_index"])[1]

        merged_preserve = dict(first)
        merged_preserve.update({
            "source": "".join([rr.get("source") or "" for rr in run]),
            "source_index": [s_start, s_end],
            "target": "".join([rr.get("target") or "" for rr in run]),
            "target_index": [t_start, t_end],
        })

        out2.append(merged_preserve)
        k = k2

    final_df = pd.DataFrame(out2, columns=merged.columns).reset_index(drop=True)
    final_df["operation_id"] = list(range(1, len(final_df) + 1))
    return final_df
