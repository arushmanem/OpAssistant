"""Retrieval metrics for the RAG eval harness.

Given a question and its expected chunk substrings, measures whether
retrieval surfaced the right chunks and how highly they were ranked.
"""
import re


def normalize(text: str) -> str:
    """Collapse all whitespace runs (spaces, newlines, tabs) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_contains_substring(chunk: str, substring: str) -> bool:
    """Case-insensitive substring check after whitespace normalization."""
    return normalize(substring).lower() in normalize(chunk).lower()


def find_first_hit_rank(chunks: list[str], substrings: list[str]) -> int | None:
    """
    Return 1-based rank of the first chunk containing ANY expected substring.
    Returns None if no chunk matches.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk_contains_substring(chunk, s) for s in substrings):
            return rank
    return None


def evaluate_question_retrieval(
    retrieved_chunks: list[str],
    expected_substrings: list[str] | None,
    match_mode: str = "any",
) -> dict:
    """
    Evaluate retrieval for a single question.

    Returns a dict with:
      - 'hit': bool, whether retrieval succeeded (based on match_mode)
      - 'rank': int or None, position of first matching chunk
      - 'reciprocal_rank': float, 1/rank if hit else 0
      - 'skipped': bool, True for adversarial questions (no expected substrings)
    """
    if expected_substrings is None:
        return {"hit": None, "rank": None, "reciprocal_rank": None, "skipped": True}

    rank = find_first_hit_rank(retrieved_chunks, expected_substrings)

    if match_mode == "all":
        all_found = all(
            any(chunk_contains_substring(c, s) for c in retrieved_chunks)
            for s in expected_substrings
        )
        hit = all_found
    else:  # "any"
        hit = rank is not None

    reciprocal_rank = 1.0 / rank if rank else 0.0

    return {
        "hit": hit,
        "rank": rank,
        "reciprocal_rank": reciprocal_rank,
        "skipped": False,
    }


def aggregate_retrieval_metrics(per_question_results: list[dict]) -> dict:
    """Average metrics across all non-skipped questions."""
    scored = [r for r in per_question_results if not r["skipped"]]
    if not scored:
        return {"recall": 0.0, "mrr": 0.0, "num_evaluated": 0}

    recall = sum(1 for r in scored if r["hit"]) / len(scored)
    mrr = sum(r["reciprocal_rank"] for r in scored) / len(scored)

    return {
        "recall": recall,
        "mrr": mrr,
        "num_evaluated": len(scored),
    }