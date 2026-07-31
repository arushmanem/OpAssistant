"""Quick sanity checks for retrieval metrics."""
from evaluation.retrieval_metrics import (
    normalize,
    chunk_contains_substring,
    find_first_hit_rank,
    evaluate_question_retrieval,
    aggregate_retrieval_metrics,
)


# Test 1: normalization handles pypdf newlines
assert normalize("Random\n \nForest") == "Random Forest"
print("[OK] normalize collapses whitespace")

# Test 2: substring match works with weird spacing
chunk = "trained a Random\n \nForest regression model"
assert chunk_contains_substring(chunk, "Random Forest")
print("[OK] substring match works across whitespace")

# Test 3: case-insensitive
assert chunk_contains_substring("Python, OCaml, SQL", "python")
print("[OK] substring match is case-insensitive")

# Test 4: first hit rank
chunks = [
    "irrelevant chunk 1",
    "irrelevant chunk 2",
    "This mentions Random Forest here",
]
assert find_first_hit_rank(chunks, ["Random Forest"]) == 3
assert find_first_hit_rank(chunks, ["nonexistent"]) is None
print("[OK] find_first_hit_rank returns correct position")

# Test 5: adversarial questions get skipped
result = evaluate_question_retrieval([], None, "any")
assert result["skipped"] is True
print("[OK] adversarial questions are skipped")

# Test 6: "all" mode requires every substring
chunks = ["chunk about UHG", "chunk about HealthPartners"]
result_all = evaluate_question_retrieval(
    chunks, ["UHG", "HealthPartners"], match_mode="all"
)
assert result_all["hit"] is True

result_all_miss = evaluate_question_retrieval(
    chunks, ["UHG", "Google"], match_mode="all"
)
assert result_all_miss["hit"] is False
print("[OK] 'all' mode requires all substrings")

# Test 7: aggregation
results = [
    {"hit": True, "rank": 1, "reciprocal_rank": 1.0, "skipped": False},
    {"hit": True, "rank": 2, "reciprocal_rank": 0.5, "skipped": False},
    {"hit": False, "rank": None, "reciprocal_rank": 0.0, "skipped": False},
    {"hit": None, "rank": None, "reciprocal_rank": None, "skipped": True},
]
agg = aggregate_retrieval_metrics(results)
assert agg["recall"] == 2 / 3
assert abs(agg["mrr"] - 0.5) < 0.001
assert agg["num_evaluated"] == 3
print("[OK] aggregation math is correct")

print("\n✅ All retrieval metric tests passed.")