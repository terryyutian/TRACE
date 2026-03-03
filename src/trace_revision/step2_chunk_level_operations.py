from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd

from .general import ensure_span_list, safe_cell

# ============================================================
# Label helpers
# ============================================================

def _is_c(label: str) -> bool:
    return isinstance(label, str) and label.startswith("C")

def _is_d(label: str) -> bool:
    return isinstance(label, str) and label.startswith("D")

def _is_i(label: str) -> bool:
    return isinstance(label, str) and label.startswith("I")

def _c_index(label: str) -> int:
    return int(label[1:])


# ============================================================
# Normalization utilities (shared format guarantees)
# ============================================================

def _normalize_op_df(op_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure source_index/target_index are ALWAYS Python lists [start,end] or None.
    Normalize NaNs in label/chunk columns to None.
    """
    op_df = op_df.copy()

    for col in ["source_index", "target_index"]:
        if col in op_df.columns:
            op_df[col] = op_df[col].apply(ensure_span_list)

    for col in ["source_label", "source_chunk", "target_label", "target_chunk"]:
        if col in op_df.columns:
            op_df[col] = op_df[col].apply(safe_cell)

    return op_df


# ============================================================
# LCS (stable Cs) with deterministic tie-break
# ============================================================

def _lcs_prefer_later(source_c: List[str], target_c: List[str]) -> List[str]:
    """
    LCS with deterministic tie-break:
      1) maximize length
      2) if tie, maximize sum of source C indices (prefer keeping later Cs stable)
    Returns one preferred LCS.
    """
    m, n = len(source_c), len(target_c)
    dp_len = [[0] * (n + 1) for _ in range(m + 1)]
    dp_w   = [[0] * (n + 1) for _ in range(m + 1)]

    def c_weight(c_label: str) -> int:
        return int(c_label[1:])

    for i in range(m):
        for j in range(n):
            if source_c[i] == target_c[j]:
                cand_len = dp_len[i][j] + 1
                cand_w   = dp_w[i][j] + c_weight(source_c[i])

                best_len, best_w = cand_len, cand_w

                l1, w1 = dp_len[i][j + 1], dp_w[i][j + 1]
                if (l1 > best_len) or (l1 == best_len and w1 > best_w):
                    best_len, best_w = l1, w1

                l2, w2 = dp_len[i + 1][j], dp_w[i + 1][j]
                if (l2 > best_len) or (l2 == best_len and w2 > best_w):
                    best_len, best_w = l2, w2

                dp_len[i + 1][j + 1] = best_len
                dp_w[i + 1][j + 1] = best_w
            else:
                l_up, w_up = dp_len[i][j + 1], dp_w[i][j + 1]
                l_left, w_left = dp_len[i + 1][j], dp_w[i + 1][j]
                if (l_up > l_left) or (l_up == l_left and w_up > w_left):
                    dp_len[i + 1][j + 1] = l_up
                    dp_w[i + 1][j + 1] = w_up
                else:
                    dp_len[i + 1][j + 1] = l_left
                    dp_w[i + 1][j + 1] = w_left

    out: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if source_c[i - 1] == target_c[j - 1]:
            w = c_weight(source_c[i - 1])
            if (dp_len[i][j] == dp_len[i - 1][j - 1] + 1) and (dp_w[i][j] == dp_w[i - 1][j - 1] + w):
                out.append(source_c[i - 1])
                i -= 1
                j -= 1
                continue

        l1, w1 = dp_len[i - 1][j], dp_w[i - 1][j]
        l2, w2 = dp_len[i][j - 1], dp_w[i][j - 1]
        if (l1 > l2) or (l1 == l2 and w1 > w2):
            i -= 1
        else:
            j -= 1

    out.reverse()
    return out


# ============================================================
# Post-hoc Replace compaction (Delete -> Insert only)
# ============================================================

def posthoc_compact_replace(op_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace definition:
      - A "Delete" followed immediately by an "Insert" (Delete -> Insert only)
      - No other operations in between
      - Do NOT treat Insert->Delete as Replace

    Label policy (NEW):
      - Replace rows are relabeled as R1, R2, ... (set on BOTH source_label and target_label)
      - Remaining Delete rows are relabeled D1, D2, ...
      - Remaining Insert rows are relabeled I1, I2, ...
      - C labels (for Equal/Move) are preserved unchanged

    This removes the confusing "Replace rows still labeled D*" issue and makes
    D/I indices refer only to non-replace deletes/inserts.
    """
    required = [
        "id", "operation",
        "source_label", "source_chunk", "source_index",
        "target_label", "target_chunk", "target_index",
    ]
    missing = [c for c in required if c not in op_df.columns]
    if missing:
        raise ValueError(f"op_df is missing required columns: {missing}")

    op_df = _normalize_op_df(op_df)
    rows = op_df.to_dict(orient="records")

    # ---- 1) Compact Delete->Insert into Replace ----
    compacted: List[Dict[str, Any]] = []
    i = 0
    while i < len(rows):
        cur = rows[i]
        if cur["operation"] == "Delete" and i + 1 < len(rows) and rows[i + 1]["operation"] == "Insert":
            nxt = rows[i + 1]
            compacted.append({
                "id": None,
                "operation": "Replace",
                "source_label": cur["source_label"],
                "source_chunk": cur["source_chunk"],
                "source_index": cur["source_index"],
                "target_label": nxt["target_label"],
                "target_chunk": nxt["target_chunk"],
                "target_index": nxt["target_index"],
            })
            i += 2
        else:
            compacted.append(cur)
            i += 1

    # ---- 2) Renumber IDs (stable) ----
    for new_id, r in enumerate(compacted, start=1):
        r["id"] = new_id

    out_df = pd.DataFrame(compacted, columns=required)
    out_df = _normalize_op_df(out_df)

    # ---- 3) Relabel Replace/Delete/Insert operations ----
    # Preserve C labels for Equal/Move. Only rewrite D/I/R labels.
    d_count = 0
    i_count = 0
    r_count = 0

    def is_c(lbl: Any) -> bool:
        return isinstance(lbl, str) and lbl.startswith("C")

    new_source_labels: List[Optional[str]] = []
    new_target_labels: List[Optional[str]] = []

    for _, row in out_df.iterrows():
        op = row["operation"]

        s_lab = safe_cell(row["source_label"])
        t_lab = safe_cell(row["target_label"])

        if op == "Replace":
            r_count += 1
            r_lab = f"R{r_count}"
            new_source_labels.append(r_lab)
            new_target_labels.append(r_lab)

        elif op == "Delete":
            d_count += 1
            # If Delete row had a C label (shouldn't), preserve it; otherwise Dk
            if is_c(s_lab):
                new_source_labels.append(s_lab)
            else:
                new_source_labels.append(f"D{d_count}")
            # target side is None for Delete
            new_target_labels.append(None)

        elif op == "Insert":
            i_count += 1
            # If Insert row had a C label (shouldn't), preserve it; otherwise Ik
            if is_c(t_lab):
                new_target_labels.append(t_lab)
            else:
                new_target_labels.append(f"I{i_count}")
            # source side is None for Insert
            new_source_labels.append(None)

        else:
            # Equal / Move (and any future ops): preserve labels
            new_source_labels.append(s_lab)
            new_target_labels.append(t_lab)

    out_df["source_label"] = new_source_labels
    out_df["target_label"] = new_target_labels

    return _normalize_op_df(out_df)

# ============================================================
# Step 2 main (public)
# ============================================================

def step2_build_operations_dataframe(src_df: pd.DataFrame, tgt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 pipeline:
      - Operations: Equal, Move, Delete, Insert, Replace (Replace is post-hoc)
      - Matched chunks (C) are output in SOURCE order
      - Equal/Move determined by LCS of C-order sequences (stable Cs = Equal)
      - Deletes bucketed by next C in SOURCE
      - Inserts bucketed by earliest SOURCE-order C to the right in TARGET
      - Sequencing per boundary: Deletes -> Inserts -> C op
      - Tail boundary END: Deletes -> Inserts
      - Enforces span format: source_index/target_index are ALWAYS Python lists [start,end] or None.
    """
    required_cols = {"chunk", "label", "chunk_index"}
    if not required_cols.issubset(src_df.columns) or not required_cols.issubset(tgt_df.columns):
        raise ValueError("src_df and tgt_df must contain columns: chunk, label, chunk_index")

    src_df = src_df.copy()
    tgt_df = tgt_df.copy()
    src_df["chunk_index"] = src_df["chunk_index"].apply(ensure_span_list)
    tgt_df["chunk_index"] = tgt_df["chunk_index"].apply(ensure_span_list)

    # ---- 1) Validate C labels and define C order ----
    src_cs = [lab for lab in src_df["label"].tolist() if _is_c(lab)]
    tgt_cs = [lab for lab in tgt_df["label"].tolist() if _is_c(lab)]

    if set(src_cs) != set(tgt_cs):
        missing_in_tgt = sorted(set(src_cs) - set(tgt_cs), key=_c_index)
        missing_in_src = sorted(set(tgt_cs) - set(src_cs), key=_c_index)
        raise ValueError(
            "C-label mismatch between src_df and tgt_df.\n"
            f"Missing in tgt: {missing_in_tgt}\n"
            f"Missing in src: {missing_in_src}"
        )

    c_labels = sorted(set(src_cs), key=_c_index)  # source order
    tgt_c_order = [lab for lab in tgt_df["label"].tolist() if _is_c(lab)]  # target appearance order

    # ---- 2) Equal vs Move via LCS ----
    stable_cs = set(_lcs_prefer_later(c_labels, tgt_c_order))

    # ---- 3) Lookup C chunk text/spans in each df ----
    src_c_info: Dict[str, Dict[str, Any]] = {}
    tgt_c_info: Dict[str, Dict[str, Any]] = {}

    for _, row in src_df.iterrows():
        lab = row["label"]
        if _is_c(lab):
            src_c_info[lab] = {"chunk": row["chunk"], "chunk_index": row["chunk_index"]}

    for _, row in tgt_df.iterrows():
        lab = row["label"]
        if _is_c(lab):
            tgt_c_info[lab] = {"chunk": row["chunk"], "chunk_index": row["chunk_index"]}

    # ---- 4) Bucket Deletes: before next C in SOURCE (or END) ----
    del_before: Dict[Optional[str], List[Dict[str, Any]]] = {lab: [] for lab in c_labels}
    del_before[None] = []

    src_labels = src_df["label"].tolist()
    for pos, row in src_df.reset_index(drop=True).iterrows():
        lab = row["label"]
        if not _is_d(lab):
            continue

        next_c: Optional[str] = None
        for q in range(pos + 1, len(src_labels)):
            if _is_c(src_labels[q]):
                next_c = src_labels[q]
                break

        bucket = next_c if next_c in del_before else None
        del_before[bucket].append({
            "label": lab,
            "chunk": row["chunk"],
            "chunk_index": row["chunk_index"],
            "pos": pos
        })

    # ---- 5) Bucket Inserts: before earliest SOURCE-order C to its right in TARGET (or END) ----
    ins_before: Dict[Optional[str], List[Dict[str, Any]]] = {lab: [] for lab in c_labels}
    ins_before[None] = []

    tgt_labels = tgt_df["label"].tolist()

    suffix_min_k: List[Optional[int]] = [None] * (len(tgt_labels) + 1)
    current_min: Optional[int] = None
    for p in range(len(tgt_labels) - 1, -1, -1):
        lab = tgt_labels[p]
        if _is_c(lab):
            k = _c_index(lab)
            current_min = k if current_min is None else min(current_min, k)
        suffix_min_k[p] = current_min
    suffix_min_k[len(tgt_labels)] = None

    k_to_c = {_c_index(lab): lab for lab in c_labels}

    for pos, row in tgt_df.reset_index(drop=True).iterrows():
        lab = row["label"]
        if not _is_i(lab):
            continue
        min_k_right = suffix_min_k[pos + 1] if pos + 1 <= len(tgt_labels) else None
        bucket = None if min_k_right is None else k_to_c[min_k_right]
        ins_before[bucket].append({
            "label": lab,
            "chunk": row["chunk"],
            "chunk_index": row["chunk_index"],
            "pos": pos
        })

    for key in del_before:
        del_before[key].sort(key=lambda x: x["pos"])
    for key in ins_before:
        ins_before[key].sort(key=lambda x: x["pos"])

    # ---- 6) Emit operations in finalized order ----
    ops: List[Dict[str, Any]] = []
    op_id = 0

    def emit(operation: str,
             src_lab: Optional[str], src_chunk: Optional[str], src_idx: Optional[List[int]],
             tgt_lab: Optional[str], tgt_chunk: Optional[str], tgt_idx: Optional[List[int]]) -> None:
        nonlocal op_id
        op_id += 1
        ops.append({
            "id": op_id,
            "operation": operation,
            "source_label": src_lab,
            "source_chunk": src_chunk,
            "source_index": src_idx,
            "target_label": tgt_lab,
            "target_chunk": tgt_chunk,
            "target_index": tgt_idx,
        })

    for c_lab in c_labels:
        for d in del_before.get(c_lab, []):
            emit("Delete", d["label"], d["chunk"], d["chunk_index"], None, None, None)

        for ins in ins_before.get(c_lab, []):
            emit("Insert", None, None, None, ins["label"], ins["chunk"], ins["chunk_index"])

        op = "Equal" if c_lab in stable_cs else "Move"
        emit(
            op,
            c_lab, src_c_info[c_lab]["chunk"], src_c_info[c_lab]["chunk_index"],
            c_lab, tgt_c_info[c_lab]["chunk"], tgt_c_info[c_lab]["chunk_index"],
        )

    for d in del_before.get(None, []):
        emit("Delete", d["label"], d["chunk"], d["chunk_index"], None, None, None)
    for ins in ins_before.get(None, []):
        emit("Insert", None, None, None, ins["label"], ins["chunk"], ins["chunk_index"])

    op_df = pd.DataFrame(
        ops,
        columns=[
            "id", "operation",
            "source_label", "source_chunk", "source_index",
            "target_label", "target_chunk", "target_index",
        ],
    )

    op_df = _normalize_op_df(op_df)
    op_df = posthoc_compact_replace(op_df)
    op_df = _normalize_op_df(op_df)
    return op_df
