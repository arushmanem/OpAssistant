"""Generation metrics for the RAG eval harness.

Two metrics:
  - Substring match: does the expected answer appear in the response?
  - LLM-as-judge: another LLM grades response quality 1-5

Also handles adversarial questions (the model should refuse).
"""
import re
from config import openai_client, CHAT_MODEL


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- Substring match ---

def substring_match(generated_answer: str, expected_answer: str) -> bool:
    """Case-insensitive: is the expected answer contained in the generated one?"""
    return normalize(expected_answer).lower() in normalize(generated_answer).lower()


# --- Adversarial refusal check ---

REFUSAL_PHRASES = [
    "i don't know",
    "i do not know",
    "not mentioned",
    "no information",
    "cannot find",
    "isn't in the context",
    "is not in the context",
    "not in the document",
    "no mention",
    "does not mention",
    "doesn't mention",
]

def is_refusal(generated_answer: str) -> bool:
    """Detect whether the model correctly declined to answer."""
    text = generated_answer.lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)


# --- LLM-as-judge ---

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about a resume.

Question: {question}
Expected answer: {expected}
AI's answer: {actual}

Rate the AI's answer 1 to 5:
5 = fully correct and complete
4 = correct but missing minor detail
3 = partially correct
2 = mostly incorrect
1 = completely wrong or hallucinated

Respond with ONLY a single digit (1-5). No explanation."""


def llm_judge(question: str, expected: str, actual: str) -> int:
    """Ask another LLM to grade the answer on a 1-5 scale."""
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    text = response.choices[0].message.content.strip()
    
    # Extract the first digit 1-5 from the response
    match = re.search(r"[1-5]", text)
    if match:
        return int(match.group())
    return 0  # judge failed to produce a valid score


# --- Per-question generation eval ---

def evaluate_question_generation(
    question: str,
    generated_answer: str,
    expected_answer: str,
    question_type: str,
) -> dict:
    """
    Evaluate a single question's generated answer.
    
    Returns dict with:
      - substring_hit: bool or None (None for synthesis/adversarial)
      - judge_score: int 1-5 (or 0 if judge failed)
      - refused: bool or None (None for non-adversarial)
      - refusal_correct: bool or None (whether refusal matched expectation)
    """
    result = {
        "substring_hit": None,
        "judge_score": None,
        "refused": None,
        "refusal_correct": None,
    }
    
    if question_type == "adversarial":
        # For adversarial, we want to see a refusal
        refused = is_refusal(generated_answer)
        result["refused"] = refused
        result["refusal_correct"] = refused  # correct if it refused
        # Adversarial questions don't get judged on substring or 1-5 score
        return result
    
    # For factual/extractive: substring match makes sense
    if question_type in ("factual", "extractive"):
        result["substring_hit"] = substring_match(generated_answer, expected_answer)
    
    # For all non-adversarial: LLM judge
    result["judge_score"] = llm_judge(question, expected_answer, generated_answer)
    
    return result


# --- Aggregation ---

def aggregate_generation_metrics(per_question_results: list[dict]) -> dict:
    """Average generation metrics across all questions."""
    # Substring match rate: only over questions where it applies
    substring_applicable = [r for r in per_question_results if r["substring_hit"] is not None]
    substring_rate = (
        sum(1 for r in substring_applicable if r["substring_hit"]) / len(substring_applicable)
        if substring_applicable else 0.0
    )
    
    # Judge score: only over questions that were judged
    judged = [r for r in per_question_results if r["judge_score"] is not None]
    avg_judge_score = (
        sum(r["judge_score"] for r in judged) / len(judged)
        if judged else 0.0
    )
    
    # Adversarial refusal rate
    adversarial = [r for r in per_question_results if r["refused"] is not None]
    refusal_rate = (
        sum(1 for r in adversarial if r["refusal_correct"]) / len(adversarial)
        if adversarial else 0.0
    )
    
    return {
        "substring_match_rate": substring_rate,
        "substring_num_evaluated": len(substring_applicable),
        "avg_judge_score": avg_judge_score,
        "judge_num_evaluated": len(judged),
        "adversarial_refusal_rate": refusal_rate,
        "adversarial_num_evaluated": len(adversarial),
    }