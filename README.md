# University Academic Information Assistant (RAG)

A Retrieval-Augmented Generation chatbot that answers student questions about
the Faculty of Computer Science curriculum (courses, codes, credit hours,
terms, prerequisites and unlocks). It answers **only** from the official TXT
documents in `data/` and refuses anything else.

One shared dataset, one ChromaDB index, one RAG system - it works the same
whether `data/` holds one year or all four.

## Stack
Python, LangChain, ChromaDB, SentenceTransformers (local embeddings),
Anthropic Claude (answer generation + evaluation judge).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your ANTHROPIC_API_KEY inside
```

## Prepare the data
```
data/yearN/termX.txt        # semester file (Year, Term, Courses, Total Credit Hours)
data/yearN/termY.txt
data/yearN/courses/CODE.txt # one file per course, filename = course code
```
Use the exact templates from the team guidelines (same field names and order,
UTF-8). Use `Not specified in the provided source.` for missing information
and `None` only when the source confirms there is none.

Check the data before indexing:
```bash
python ingest.py
```
It prints how many documents were loaded and warns about inconsistencies
(totals, missing course files, wrong term, filename vs code...).

## Build the index
```bash
python index.py
```
Rebuilds `chroma_db/` from scratch. Re-run it after any change in `data/`.

## Start the chatbot
```bash
python app.py
```
Type `exit` to quit.

## Run the evaluation
```bash
python evaluate.py --questions year2_questions.json
```
Writes `evaluation_results.md` and `evaluation_results.json` with the
Pass/Fail table and the overall score. After merging all years, combine every
member's questions into one JSON file (same keys: id, category, question,
expected_answer; category names start with A-G) and pass it with `--questions`.

## How retrieval works
For each question the system retrieves the **top 4** documents:
1. exact matches first - course codes and course names mentioned in the
   question, plus the term/year summary for "Term N" / "Year N";
2. semantic search (embeddings) to fill the remaining slots.

`ingest.py` also builds derived summary documents per term and per year (made
only from the course files) so questions like "list all courses in Term 3 with
prerequisites" fit inside the top-4 limit.

## Optional settings (.env)
`LLM_MODEL` (default `claude-sonnet-5-5`), `EMBED_MODEL`
(default `sentence-transformers/all-MiniLM-L6-v2`).
