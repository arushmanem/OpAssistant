"""Entry point: orchestrates the RAG pipeline."""
from rag.document_loader import load_pdf, chunk_text
from rag.embeddings import embed_batch, embed_text
from rag.vector_store import get_collection, store_chunks
from rag.hybrid_retrieval import BM25Index, hybrid_retrieve
from rag.llm import generate_answer
from config import RESUME_PATH


def ingest(pdf_path: str):
    """Load a PDF, chunk it, embed it, and store it."""
    print(f"Loading {pdf_path}...")
    text = load_pdf(pdf_path)
    
    print("Chunking...")
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks")
    
    print("Embedding...")
    embeddings = embed_batch(chunks)
    print(f"  → {len(embeddings)} vectors")
    
    print("Storing in ChromaDB...")
    collection = get_collection(reset=True)
    store_chunks(collection, chunks, embeddings)
    print(f"  → {collection.count()} chunks stored\n")

    bm25_index = BM25Index(chunks)

    return collection, bm25_index


def ask(collection, bm25_index, question: str) -> str:
    """Answer a question using the RAG pipeline."""
    question_vector = embed_text(question)
    chunks = hybrid_retrieve(collection, bm25_index, question, question_vector, n_results=5)

    # Debug: show what chunks were retrieved
    print("  [DEBUG] Retrieved chunks:")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i+1}: {c[:150]}...")
    
    return generate_answer(question, chunks)


if __name__ == "__main__":
    # Ingest the PDF once at startup
    collection, bm25_index = ingest(RESUME_PATH)
    
    print("Ready! Ask questions about the document. Type 'quit' to exit.\n")
    
    while True:
        question = input("Q: ").strip()
        
        # Exit conditions
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        
        # Skip empty input (user just hit Enter)
        if not question:
            continue
        
        # Answer and print
        answer = ask(collection, bm25_index, question)
        print(f"A: {answer}\n")