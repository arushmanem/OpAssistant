"""Read PDFs and split their text into chunks."""
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(path: str) -> str:
    """Extract all text from a PDF into a single string."""
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks