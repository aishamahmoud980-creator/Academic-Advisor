"""app.py - simple command-line chat. Type 'exit' to quit."""
from rag import AcademicRAG


def main():
    rag = AcademicRAG()
    print("University Academic Information Assistant")
    print("Ask about courses, prerequisites, terms and credit hours. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        result = rag.ask(q)
        print(f"\nAssistant: {result['answer']}\n")
        sources = ", ".join(d.metadata.get("doc_id", "?") for d in result["docs"])
        print(f"(sources: {sources})\n")
    print("Goodbye!")


if __name__ == "__main__":
    main()
