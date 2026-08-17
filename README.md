# AI Operations Assistant

A retrieval-augmented generation pipeline built from scratch, with a hybrid retriever and a real evaluation harness. No LangChain, no orchestration framework: every stage is implemented directly so the failure modes stay visible.

Ask a natural-language question about an ingested PDF and get an answer grounded in the document, with refusal when the document does not contain the answer.

## Why hybrid retrieval

Pure semantic search was the first version, and it failed in a specific way. Embeddings are strong on paraphrase but dilute rare tokens, so queries containing terms like `SHAP`, `UiPath`, or `R²` retrieved topically-adjacent chunks instead of the exact one. BM25 has the opposite profile: excellent on rare exact tokens, useless on paraphrase.

The retriever runs both and merges the rankings with Reciprocal Rank Fusion (`RRF_K = 60`, the constant from the original paper). RRF fuses on rank rather than score, so the two systems do not need comparable score scales.

## Architecture

```
PDF ─> document_loader ─> chunk (800 chars, 100 overlap)
                              │
                ┌─────────────┴─────────────┐
                v                           v
        embeddings (OpenAI            BM25Index
        text-embedding-3-small)       (rank_bm25)
                │                           │
                v                           │
          ChromaDB (persistent)             │
                │                           │
                └────────> RRF fusion <─────┘
                                │
                                v
                    llm.py (gpt-4o-mini, grounded prompt)
                                │
                                v
                             answer
```

| Module | Responsibility |
|---|---|
| `config.py` | All model names, chunk sizes, and paths in one place |
| `rag/document_loader.py` | PDF text extraction and overlapping chunking |
| `rag/embeddings.py` | Embedding calls |
| `rag/vector_store.py` | ChromaDB collection lifecycle and semantic retrieval |
| `rag/hybrid_retrieval.py` | BM25 index and RRF fusion |
| `rag/llm.py` | Prompt construction and answer generation |
| `evaluation/` | Retrieval and generation metrics, evaluation runner |
| `tests/` | Unit tests for metrics and fusion |

## Evaluation

Retrieval and generation are measured separately, because a wrong answer can come from either stage and averaging them hides which.

Dataset: 20 hand-written cases in `eval_dataset.json`, split between factual questions and adversarial questions whose answers are deliberately absent from the document.

| Metric | Result |
|---|---|
| MRR | 0.873 |
| Recall@k | see `evaluation/retrieval_metrics.py` |
| LLM-as-judge (1-5) | 4.35 → 4.65 after prompt iteration |
| Adversarial refusal rate | 100% |

The judge score moved from 4.35 to 4.65 by rewriting the prompt to require every part of a multi-part question to be addressed and to preserve exact figures rather than paraphrasing them away. The failures that drove that change were multi-entity synthesis queries, where the model would answer about one internship and silently drop the others.

## Running it

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python main.py
```

Evaluation:

```bash
python -m evaluation.evaluate
```

Tests:

```bash
python -m pytest tests/
```

## Known limitations and what I would do next

- `embed_batch` loops one call per chunk instead of batching, which is the obvious throughput fix.
- Chunking is character-level with fixed overlap. Semantic or recursive chunking would stop mid-sentence splits.
- The LLM-as-judge evaluator uses the same model family as the generator, which risks shared blind spots. A different judge model would be a cleaner test.
- 20 evaluation cases is enough to catch regressions, not enough for statistical confidence on small deltas.
- No reranker. A cross-encoder pass over the fused top-k would likely be the single largest quality gain.

## Stack

Python, OpenAI API (embeddings + chat), ChromaDB, rank_bm25, pypdf, pytest
