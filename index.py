"""
index.py
Builds ONE shared ChromaDB index from every document found by ingest.py.
Re-running it rebuilds the database from scratch.
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_documents

load_dotenv()
DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION = "academic_advisor"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index():
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found in data/. Nothing to index.")
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
        ids=[d.metadata["doc_id"] for d in docs],
    )
    print(f"Indexed {len(docs)} documents into {DB_DIR}")


if __name__ == "__main__":
    build_index()
