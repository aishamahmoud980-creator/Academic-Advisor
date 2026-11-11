"""rag.py - retrieves the top 4 chunks from ChromaDB and asks Groq to answer from them only."""
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from index import load_index

load_dotenv()  # reads GROQ_API_KEY (and optional GROQ_MODEL) from .env

TOP_K = 4
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

PROMPT = """You are a University Academic Information Assistant
for the Faculty of Computer Science.

Your job is to answer student questions about academic
courses and curriculum information.

STRICT RULES:

1. Answer using ONLY the information in the retrieved context.

2. Do not use outside knowledge or make assumptions.

3. Do not invent course codes, course names, credit hours,
   prerequisites, descriptions, or semester information.

4. If the requested information is not available in the
   retrieved context, respond exactly:

   "I couldn't find this information in the provided
   academic documents."

5. Always mention the course code when answering
   a course-related question, whenever it is available.

6. Preserve AND/OR conditions in prerequisites.

7. When explaining course progression, do not claim that
   a course is fully available if other prerequisites
   are still required.

8. If a question asks for an opinion or information
   unrelated to the provided academic documents,
   politely explain that you only answer questions
   supported by the academic documents.

9. If a question is ambiguous, ask the student to clarify
   instead of guessing.

10. Keep answers clear, organized, and student-friendly.

RETRIEVED CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:"""

_db = None
_llm = None


def _get_db():
    global _db
    if _db is None:
        _db = load_index()
    return _db


def _get_llm():
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found. Add it to the .env file.")
        _llm = ChatGroq(model=MODEL_NAME, temperature=0)
    return _llm


def codes_in_question(question: str) -> list[str]:
    """Course codes written in the question, e.g. CCS1302 (case-insensitive)."""
    codes = []
    for code in re.findall(r"\b[A-Za-z]{3}\d{4}\b", question):
        code = code.upper()
        if code not in codes:
            codes.append(code)
    return codes


def terms_in_question(question: str) -> list[int]:
    """Terms the question talks about: 'Term 2', 'Year 1' (= terms 1-2), 'previous term'."""
    q = question.lower()
    terms = {int(n) for n in re.findall(r"\bterms?\s*(\d+)", q)}
    for n in re.findall(r"\byears?\s*(\d+)", q):
        y = int(n)
        terms.update({2 * y - 1, 2 * y})
    if "previous term" in q or "prior term" in q or "preceding term" in q:
        terms.update({t - 1 for t in list(terms) if t > 1})
    return sorted(terms)


def retrieve(question: str):
    """Top 4 chunks across ALL years.

    1. Course files for any course code written in the question (exact match).
    2. Semester files for any term/year named in the question.
    3. The remaining slots are filled by vector similarity.
    The total is always at most 4 chunks.
    """
    db = _get_db()
    docs, seen = [], set()

    def add(meta, text):
        if len(docs) < TOP_K and meta["source"] not in seen:
            docs.append(Document(page_content=text, metadata=meta))
            seen.add(meta["source"])

    codes = codes_in_question(question)
    if codes:
        found = db.get(where={"course_code": {"$in": codes}}, include=["documents", "metadatas"])
        for meta, text in sorted(zip(found["metadatas"], found["documents"]),
                                 key=lambda p: codes.index(p[0]["course_code"])):
            add(meta, text)

    terms = terms_in_question(question)
    if terms:
        found = db.get(
            where={"$and": [{"doc_type": "term"}, {"term": {"$in": terms}}]},
            include=["documents", "metadatas"],
        )
        for meta, text in sorted(zip(found["metadatas"], found["documents"]),
                                 key=lambda p: p[0]["term"]):
            add(meta, text)

    for doc in db.similarity_search(question, k=TOP_K):
        add(doc.metadata, doc.page_content)
    return docs


def ask(question: str) -> dict:
    """Return {'answer': str, 'sources': [file paths]}."""
    docs = retrieve(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = PROMPT.format(context=context, question=question)
    response = _get_llm().invoke(prompt)
    return {
        "answer": response.content.strip(),
        "sources": [d.metadata.get("source", "?") for d in docs],
    }


if __name__ == "__main__":
    result = ask("What are the prerequisites for Calculus II?")
    print(result["answer"])
    print("\nSources:", result["sources"])
