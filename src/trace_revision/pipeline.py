from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import pandas as pd

from .step1_chunk_matching import step1_chunk_matching_to_dataframes, build_spacy_nlp
from .step2_chunk_level_operations import step2_build_operations_dataframe
from .step3_token_level_analysis import step3_fine_grained_revision_analysis
from .step3_token_level_analysis import refine_step3_merge_adjacent_chunks_by_label


def trace(
    source_text: str,
    target_text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    # Step 1
    min_sent_tokens: int = 4,
    theta: float = 0.6,
    delta_len: int = 5,
    strict_increase: bool = True,
    group_unmatched: bool = False,
    group_adjacent_unmatched_posthoc: bool = True,
    # Step 3
    replace_similarity_threshold: float = 0.4,
    # Step3 readability refinement
    refine_sentence_units: bool = True,
    # Debug
    debug: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Run TRACE end-to-end and return:
      - step3_df (final mixed-granularity revision representation)
      - step3_dict (list of row dictionaries; same content as step3_df)
      - debug_info (optional)
    """
    nlp = build_spacy_nlp(spacy_model)

    src_df, tgt_df, dbg1 = step1_chunk_matching_to_dataframes(
        source_text,
        target_text,
        nlp=nlp,
        min_sent_tokens=min_sent_tokens,
        theta=theta,
        delta_len=delta_len,
        strict_increase=strict_increase,
        group_unmatched=group_unmatched,
        group_adjacent_unmatched_posthoc=group_adjacent_unmatched_posthoc,
        debug=debug,
    )

    step2_df = step2_build_operations_dataframe(src_df, tgt_df)

    step3_df = step3_fine_grained_revision_analysis(
        step2_df,
        replace_similarity_threshold=replace_similarity_threshold,
    )

    if refine_sentence_units:
       step3_df = refine_step3_merge_adjacent_chunks_by_label(step3_df)

    step3_dict = step3_df.to_dict(orient="records")

    debug_info = None
    if debug:
        debug_info = {
            "step1": dbg1,
            "step2_df": step2_df,
            "src_df": src_df,
            "tgt_df": tgt_df,
            "params": {
                "spacy_model": spacy_model,
                "min_sent_tokens": min_sent_tokens,
                "theta": theta,
                "delta_len": delta_len,
                "strict_increase": strict_increase,
                "group_unmatched": group_unmatched,
                "group_adjacent_unmatched_posthoc": group_adjacent_unmatched_posthoc,
                "replace_similarity_threshold": replace_similarity_threshold,
            },
        }

    return step3_df, step3_dict, debug_info


def _visible_newlines_in_cell(x):
    """
    Replace literal newline characters with the visible marker '<newline>'
    for display/export only. Non-string values are returned unchanged.
    """
    if isinstance(x, str):
        return x.replace("\n", "<newline>")
    return x

def make_newlines_visible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where all string cells have '\n' replaced by
    '<newline>' for easier CSV/display inspection.

    This is a DISPLAY/EXPORT helper only. It does not change analysis spans.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(_visible_newlines_in_cell)
    return out