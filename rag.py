"""
rag.py
Retrieves the top 4 chunks and asks the LLM to answer strictly from them.

Retrieval = exact matches first (course codes / course names / "Term N" /
"Year N" mentioned in the question), then semantic search to fill the 4 slots.

The LLM is chosen from .env as an ordered list (LLM_PROVIDERS=groq,google,ollama).
The first provider is tried first; on a temporary error (503 / 429) it retries
a couple of times, then moves on to the next provider in the list.
"""
import os
import re
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from index import COLLECTION, DB_DIR, get_embeddings

load_dotenv()
TOP_K = 4
REFUSAL = "I couldn't find this information in the provided academic documents."
BUSY_MESSAGE = ("The AI server is busy right now. "
                "Please ask your question again in a few seconds.")

# retries per provider on a temporary error, and the base wait (seconds)
RETRY_ATTEMPTS = 3
RETRY_BASE_WAIT = 3  # waits 3, 6 seconds between attempts

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


def _norm(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def _is_temporary_error(e: Exception) -> bool:
    """True for server-busy / rate-limit errors that are worth retrying."""
    msg = str(e).upper()
    return any(s in msg for s in ("503", "UNAVAILABLE", "OVERLOADED",
                                  "429", "RESOURCE_EXHAUSTED", "RATE LIMIT"))


DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "google": "gemini-3.5-flash",
    "ollama": "llama3.1",
    "anthropic": "claude-sonnet-5-5",
}


def make_llm(provider: str):
    """Build one LLM. Model name comes from <PROVIDER>_MODEL in .env (or a default)."""
    provider = provider.strip().lower()
    model = os.getenv(f"{provider.upper()}_MODEL", "").strip() or DEFAULT_MODELS.get(provider, "")
    if provider == "groq":        # GROQ_API_KEY from console.groq.com
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, max_retries=0)
    if provider == "google":      # GOOGLE_API_KEY from Google AI Studio
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0)
    if provider == "ollama":      # local, no key
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    if provider == "anthropic":   # ANTHROPIC_API_KEY (paid)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)
    raise ValueError(f"Unknown provider: {provider}")


def build_providers():
    """Ordered list of (name, llm). Providers that fail to load are skipped."""
    names = os.getenv("LLM_PROVIDERS") or os.getenv("LLM_PROVIDER", "groq")
    out = []
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        try:
            out.append((name, make_llm(name)))
        except Exception as e:
            print(f"  [skipping provider '{name}': {type(e).__name__}: {e}]")
    if not out:
        raise RuntimeError("No LLM provider could be loaded. Check your .env file.")
    return out


class AcademicRAG:
    def __init__(self, k: int = TOP_K):
        if not DB_DIR.exists():
            raise FileNotFoundError("chroma_db not found. Run `python index.py` first.")
        self.k = k
        self.db = Chroma(collection_name=COLLECTION,
                         embedding_function=get_embeddings(),
                         persist_directory=str(DB_DIR))
        self.prompt = ChatPromptTemplate.from_template(PROMPT)
        self.chains = [(name, self.prompt | llm | StrOutputParser())
                       for name, llm in build_providers()]
        print("LLM order:", " -> ".join(n for n, _ in self.chains))

        res = self.db.get(where={"doc_type": "course"}, include=["metadatas"])
        self._names = [(_norm(m["course_name"]), m["course_code"])
                       for m in res["metadatas"]]

    # ---- helpers
    def _to_docs(self, res):
        return [Document(page_content=t, metadata=m)
                for t, m in zip(res["documents"], res["metadatas"])]

    def _fetch(self, where=None, ids=None):
        return self._to_docs(self.db.get(where=where, ids=ids,
                                         include=["documents", "metadatas"]))

    # ---- retrieval
    def retrieve(self, question: str):
        picked, seen = [], set()

        def add(docs):
            for d in docs:
                did = d.metadata.get("doc_id")
                if did not in seen:
                    seen.add(did)
                    picked.append(d)

        q_norm, q_low = _norm(question), question.lower()
        codes = list(dict.fromkeys(re.findall(r"[A-Z]{2,4}\d{3,4}", question.upper())))
        for name, code in self._names:
            if name in q_norm and code not in codes:
                codes.append(code)
        for code in codes:
            add(self._fetch(ids=[f"course::{code}"]))
        for t in dict.fromkeys(re.findall(r"term\s*(\d+)", q_low)):
            add(self._fetch(where={"$and": [{"doc_type": "term_summary"},
                                            {"term": int(t)}]}))
        for y in dict.fromkeys(re.findall(r"year\s*(\d+)", q_low)):
            add(self._fetch(where={"$and": [{"doc_type": "year_summary"},
                                            {"year": int(y)}]}))
        picked = picked[: self.k]
        if len(picked) < self.k:
            add(self.db.similarity_search(question, k=self.k * 2))
        return picked[: self.k]

    # ---- LLM call: retry, then fall through the provider list
    def _invoke(self, inputs: dict):
        """Returns (answer_text, ok). Never raises; falls back to the next provider."""
        for name, chain in self.chains:
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    return chain.invoke(inputs), True
                except Exception as e:
                    if _is_temporary_error(e) and attempt < RETRY_ATTEMPTS:
                        wait = RETRY_BASE_WAIT * attempt
                        print(f"  [{name} busy, retrying in {wait}s...]")
                        time.sleep(wait)
                        continue
                    print(f"  [{name} failed: {type(e).__name__}: {str(e)[:120]}]")
                    break  # go to the next provider
        return BUSY_MESSAGE, False

    def ask(self, question: str) -> dict:
        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer, ok = self._invoke({"context": context, "question": question})
        return {"answer": answer.strip(), "docs": docs, "ok": ok}