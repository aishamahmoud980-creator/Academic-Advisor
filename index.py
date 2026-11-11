"""index.py - builds ONE shared ChromaDB vector database from all TXT files."""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_documents, validate

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "academic_advisor"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Same embedding model must be used for indexing and for searching."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index() -> Chroma:
    problems = validate()
    if problems:
        print("Fix these data problems first:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    documents = load_documents()
    print(f"Indexing {len(documents)} documents...")

    # Rebuild from scratch so re-running never creates duplicates.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    # Each file is small (one course or one semester), so no splitting is needed.
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )


def load_index() -> Chroma:
    """Open the existing database (used by rag.py)."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    db = build_index()
    print("Index built. Quick test search: 'prerequisite of Calculus II'")
    for doc in db.similarity_search("prerequisite of Calculus II", k=4):
        print(" -", doc.metadata["source"])
