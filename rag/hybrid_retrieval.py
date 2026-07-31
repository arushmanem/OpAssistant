"""Hybrid retrieval: BM25 keyword search fused with semantic search via RRF.

Semantic search is strong on paraphrase but dilutes rare tokens (SHAP, UiPath, R²).
BM25 is the opposite. Reciprocal Rank Fusion merges both rankings without needing
their scores to be on a comparable scale.
"""
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from config import RRF_K


def tokenize(text: str) -> list[str]:
    """Lowercase and pull out alphanumeric runs.

    Extracting runs (rather than splitting on whitespace) means the odd spacing
    pypdf leaves behind doesn't produce junk tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Keyword index over the same chunk list that's stored in ChromaDB.

    Chunk positions here must line up with the `chunk_{i}` ids in vector_store,
    since that's how the two rankings get matched up during fusion.
    """

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])

    def search(self, query: str, n_results: int = 5) -> list[int]:
        """Return chunk indices for the top-N keyword matches, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return ranked[:n_results]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked lists of chunk indices into one ranking.

    Each list contributes 1/(k + rank) to every chunk it ranks, so a chunk both
    retrievers like beats one that only a single retriever ranked highly. The k
    term flattens the curve near the top, keeping rank 1 from dominating.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            scores[index] += 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def semantic_search(collection, question_vector: list[float], n_results: int) -> list[int]:
    """Return chunk indices for the top-N nearest vectors, best first."""
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(n_results, collection.count()),
    )
    # ids come back as "chunk_{i}" — the format store_chunks writes.
    return [int(chunk_id.split("_")[1]) for chunk_id in results["ids"][0]]


def hybrid_retrieve(
    collection,
    bm25_index: BM25Index,
    question: str,
    question_vector: list[float],
    n_results: int = 5,
    candidate_pool: int | None = None,
) -> list[str]:
    """Retrieve chunks by fusing semantic and keyword rankings.

    Each retriever proposes `candidate_pool` chunks so fusion has room to
    reorder; only the top `n_results` survive.
    """
    if candidate_pool is None:
        candidate_pool = max(10, n_results * 2)

    semantic_ranking = semantic_search(collection, question_vector, candidate_pool)
    keyword_ranking = bm25_index.search(question, candidate_pool)

    fused = reciprocal_rank_fusion([semantic_ranking, keyword_ranking])
    return [bm25_index.chunks[i] for i in fused[:n_results]]
