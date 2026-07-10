"""Turn text into vectors using OpenAI's embedding API."""
from config import openai_client, EMBEDDING_MODEL


def embed_text(text: str) -> list[float]:
    """Turn a single string into a vector."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings. Just loops for now, could batch later."""
    return [embed_text(t) for t in texts]