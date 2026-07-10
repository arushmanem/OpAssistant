"""Build prompts and generate answers with the LLM."""
from config import openai_client, CHAT_MODEL


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a resume.
Use ONLY the context below to answer the question. If the answer isn't in the context, say you don't know.

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
    )
    return response.choices[0].message.content