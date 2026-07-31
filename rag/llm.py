"""Build prompts and generate answers with the LLM."""
from config import openai_client, CHAT_MODEL


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.

Use ONLY the context below to answer. Follow these rules:

1. If the question asks about MULTIPLE things (e.g., "What internships has he had?", "What projects has he built?"), address EACH item completely. Don't leave any out.

2. Preserve specific details from the context: numbers, dates, metrics, technical terms, company names, tool names. If the context says "R² = 0.88", include that; don't paraphrase it away.

3. For "what is X?" or "who is X?" questions, START with a brief identity statement (one sentence saying what X IS), THEN add details. Example: "SnapCount is an NFL Fantasy Analytics Platform. It uses Python, React..." not just a list of features.

4. For list-style questions, format your answer as a bulleted or numbered list so nothing gets lost.

5. If the answer is NOT in the context, respond with exactly: "I don't know based on the document provided." Do not guess or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """Given a question and relevant chunks, ask the LLM for a grounded answer."""
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,  # generous, prevents truncation on list answers
    )
    return response.choices[0].message.content