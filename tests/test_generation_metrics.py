"""Sanity checks for generation metrics.

These don't call the real LLM — they test the pure functions only.
"""
from evaluation.generation_metrics import (
    substring_match,
    is_refusal,
    aggregate_generation_metrics,
)


# Test 1: substring match is case-insensitive and whitespace-tolerant
assert substring_match("Arush has a GPA of 3.85", "3.85")
assert substring_match("His majors are Data Science and Computer Science.", "data science and computer science")
assert not substring_match("Arush studies engineering", "3.85")
print("[OK] substring_match works")

# Test 2: refusal detection
assert is_refusal("I don't know.")
assert is_refusal("The document does not mention this.")
assert is_refusal("This isn't in the context provided.")
assert not is_refusal("Arush graduated in May 2027.")
print("[OK] is_refusal detects known refusal phrases")

# Test 3: aggregation with mixed question types
results = [
    # 2 factual: one hit, one miss
    {"substring_hit": True,  "judge_score": 5, "refused": None, "refusal_correct": None},
    {"substring_hit": False, "judge_score": 2, "refused": None, "refusal_correct": None},
    # 1 synthesis: no substring, but judged
    {"substring_hit": None,  "judge_score": 4, "refused": None, "refusal_correct": None},
    # 2 adversarial: one correct refusal, one hallucination
    {"substring_hit": None,  "judge_score": None, "refused": True,  "refusal_correct": True},
    {"substring_hit": None,  "judge_score": None, "refused": False, "refusal_correct": False},
]
agg = aggregate_generation_metrics(results)
assert agg["substring_match_rate"] == 0.5  # 1 of 2 factual hit
assert agg["substring_num_evaluated"] == 2
assert abs(agg["avg_judge_score"] - (5 + 2 + 4) / 3) < 0.001
assert agg["judge_num_evaluated"] == 3
assert agg["adversarial_refusal_rate"] == 0.5  # 1 of 2 adversarial refused
assert agg["adversarial_num_evaluated"] == 2
print("[OK] aggregation math is correct")

print("\n✅ All generation metric tests passed.")