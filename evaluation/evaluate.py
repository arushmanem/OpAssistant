"""Full eval harness: runs the RAG pipeline on the eval dataset and reports quality.

Usage:
    python -m evaluation.evaluate
"""
import json
from config import CHROMA_PATH

from rag.document_loader import load_pdf, chunk_text
from rag.embeddings import embed_batch, embed_text
from rag.vector_store import get_collection, store_chunks
from rag.hybrid_retrieval import BM25Index, hybrid_retrieve
from rag.llm import generate_answer
from config import RESUME_PATH

from evaluation.retrieval_metrics import (
    evaluate_question_retrieval,
    aggregate_retrieval_metrics,
)
from evaluation.generation_metrics import (
    evaluate_question_generation,
    aggregate_generation_metrics,
)


# --- Config ---

EVAL_DATASET_PATH = "eval_dataset.json"
N_RESULTS = 5  # top-k for retrieval


# --- Setup ---

def setup_collection():
    """Fresh ingestion of the resume so eval runs are deterministic."""
    print(f"Ingesting {RESUME_PATH}...")
    text = load_pdf(RESUME_PATH)
    chunks = chunk_text(text)
    embeddings = embed_batch(chunks)
    collection = get_collection(reset=True)
    store_chunks(collection, chunks, embeddings)
    bm25_index = BM25Index(chunks)
    print(f"  → {collection.count()} chunks stored\n")
    return collection, bm25_index


# --- Per-question eval ---

def run_one_question(collection, bm25_index, question_item: dict) -> dict:
    """Run the pipeline on one question and return combined metrics."""
    question = question_item["question"]
    question_type = question_item["type"]
    expected_answer = question_item["expected_answer"]
    expected_substrings = question_item["expected_chunk_substrings"]
    match_mode = question_item.get("match_mode", "any")

    # Retrieval
    question_vector = embed_text(question)
    retrieved_chunks = hybrid_retrieve(
        collection, bm25_index, question, question_vector, n_results=N_RESULTS
    )

    retrieval = evaluate_question_retrieval(
        retrieved_chunks, expected_substrings, match_mode
    )
    
    # Generation
    generated_answer = generate_answer(question, retrieved_chunks)
    
    generation = evaluate_question_generation(
        question, generated_answer, expected_answer, question_type
    )
    
    return {
        "question": question,
        "type": question_type,
        "generated_answer": generated_answer,
        "retrieval": retrieval,
        "generation": generation,
    }


# --- Reporting ---

def print_question_status(i: int, total: int, result: dict):
    """One line per question showing pass/fail flags."""
    q = result["question"]
    r = result["retrieval"]
    g = result["generation"]
    
    if r["skipped"]:
        retrieval_flag = "  --  "
    else:
        retrieval_flag = "  R:OK" if r["hit"] else "R:MISS"
    
    if g["substring_hit"] is True:
        substring_flag = "S:OK  "
    elif g["substring_hit"] is False:
        substring_flag = "S:MISS"
    else:
        substring_flag = "  --  "
    
    if g["judge_score"]:
        judge_flag = f"J:{g['judge_score']}"
    else:
        judge_flag = "  --"
    
    if g["refused"] is True:
        adv_flag = "A:OK"
    elif g["refused"] is False:
        adv_flag = "A:MISS"
    else:
        adv_flag = "  --"
    
    print(f"  [{i:2d}/{total}] {retrieval_flag} {substring_flag} {judge_flag} {adv_flag}  {q[:60]}")


def print_report(all_results: list[dict]):
    """Print the final aggregated report."""
    retrieval_results = [r["retrieval"] for r in all_results]
    generation_results = [r["generation"] for r in all_results]
    
    retrieval_agg = aggregate_retrieval_metrics(retrieval_results)
    generation_agg = aggregate_generation_metrics(generation_results)
    
    print("\n" + "=" * 60)
    print("FINAL EVAL REPORT")
    print("=" * 60)
    
    print("\nRetrieval metrics (top-{})".format(N_RESULTS))
    print(f"  Recall:     {retrieval_agg['recall']:.2%}")
    print(f"  MRR:        {retrieval_agg['mrr']:.3f}")
    print(f"  Evaluated:  {retrieval_agg['num_evaluated']} questions")
    
    print("\nGeneration metrics")
    print(f"  Substring match rate:   {generation_agg['substring_match_rate']:.2%} ({generation_agg['substring_num_evaluated']} questions)")
    print(f"  Avg LLM-judge score:    {generation_agg['avg_judge_score']:.2f} / 5.00 ({generation_agg['judge_num_evaluated']} questions)")
    print(f"  Adversarial refusal:    {generation_agg['adversarial_refusal_rate']:.2%} ({generation_agg['adversarial_num_evaluated']} questions)")
    
    # Failure log
    print("\nFailures worth reviewing:")
    any_failure = False
    for result in all_results:
        q = result["question"]
        r = result["retrieval"]
        g = result["generation"]
        
        problems = []
        if r["hit"] is False:
            problems.append("retrieval miss")
        if g["substring_hit"] is False:
            problems.append("substring miss")
        if g["judge_score"] and g["judge_score"] <= 3:
            problems.append(f"low judge score ({g['judge_score']})")
        if g["refused"] is False:
            problems.append("hallucinated on adversarial")
        
        if problems:
            any_failure = True
            print(f"  - Q: {q}")
            print(f"    Issues: {', '.join(problems)}")
            print(f"    Answer: {result['generated_answer'][:150]}...")
    
    if not any_failure:
        print("  (none — everything looks clean)")
    
    print("=" * 60)


# --- Main ---

def main():
    with open(EVAL_DATASET_PATH) as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} eval questions.\n")
    
    collection, bm25_index = setup_collection()

    print("Running eval...")
    print(f"  Legend: R=retrieval  S=substring  J=judge(1-5)  A=adversarial")
    print()
    
    all_results = []
    for i, item in enumerate(dataset, start=1):
        result = run_one_question(collection, bm25_index, item)
        print_question_status(i, len(dataset), result)
        all_results.append(result)
    
    print_report(all_results)


if __name__ == "__main__":
    main()