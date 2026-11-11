"""app.py - simple command-line chat for the University Academic Information Assistant."""
from rag import ask

EXIT_WORDS = {"exit", "quit", "q", "bye", "خروج"}
SHOW_SOURCES = True  # set to False to hide the source files under each answer


def main():
    print("=" * 60)
    print(" University Academic Information Assistant")
    print(" Ask about courses, credit hours, terms, prerequisites.")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        try:
            result = ask(question)
        except Exception as error:  # e.g. no internet, wrong API key, rate limit
            print(f"\nSomething went wrong: {error}")
            continue

        print(f"\nAssistant: {result['answer']}")
        if SHOW_SOURCES:
            print("\n[Sources: " + ", ".join(result["sources"]) + "]")


if __name__ == "__main__":
    main()
