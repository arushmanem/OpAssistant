"""Centralized configuration: env vars and API clients."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Verify the API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# Shared OpenAI client (used by embeddings.py and llm.py)
openai_client = OpenAI()

# Model choices in one place so they're easy to swap
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ChromaDB config
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "resume"