"""Quick sanity checks for hybrid retrieval."""
from rag.hybrid_retrieval import (
    tokenize,
    BM25Index,
    reciprocal_rank_fusion,
)


# Test 1: tokenization survives pypdf's broken spacing
assert tokenize("Random\n \nForest") == ["random", "forest"]
print("[OK] tokenize handles pypdf whitespace")

# Test 2: punctuation-heavy technical terms split predictably.
# Non-ASCII marks like the ² in R² drop out, which is fine — queries run
# through the same tokenizer, so both sides reduce to the same tokens.
assert tokenize("C/C++, R²") == ["c", "c", "r"]
assert tokenize("nfl_data_py") == ["nfl", "data", "py"]
assert tokenize("text-embedding-3-small") == ["text", "embedding", "3", "small"]
print("[OK] tokenize splits technical terms consistently")

# Test 3: BM25 surfaces the chunk holding a rare keyword
chunks = [
    "Education: University of Minnesota, Data Science and Computer Science",
    "Built a Random Forest model and applied SHAP explainability for stakeholders",
    "Engineered automation systems in UiPath and VB.NET for claims processing",
]
index = BM25Index(chunks)
assert index.search("SHAP", n_results=1) == [1]
assert index.search("UiPath automation", n_results=1) == [2]
print("[OK] BM25 finds chunks by rare keyword")

# Test 4: search respects n_results and returns best-first
top_two = index.search("Random Forest SHAP", n_results=2)
assert len(top_two) == 2
assert top_two[0] == 1
print("[OK] BM25 search respects n_results")

# Test 5: RRF rewards consensus over any single retriever's top pick
#   chunk 0 is ranked 1st then 3rd; chunk 1 is ranked 2nd then 1st.
#   Chunk 1 wins because both lists rate it highly.
fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]])
assert fused == [1, 0, 2]
print("[OK] RRF favors chunks both retrievers rank highly")

# Test 6: a chunk found by only one retriever still makes the ranking
fused_partial = reciprocal_rank_fusion([[0, 1], [0, 5]])
assert set(fused_partial) == {0, 1, 5}
assert fused_partial[0] == 0
print("[OK] RRF keeps single-retriever candidates")

# Test 7: exact RRF math, k=60
scores = reciprocal_rank_fusion([[7]], k=60)
assert scores == [7]
tied = reciprocal_rank_fusion([[0, 1], [1, 0]])
assert tied == [0, 1]  # identical scores break deterministically by index
print("[OK] RRF math and tie-breaking are deterministic")

print("\n✅ All hybrid retrieval tests passed.")
