"""evaluate.py - runs the testing questions through the RAG system and scores the answers.

Question files: every file named questions*.json in this folder is loaded
(e.g. questions_year1.json ... questions_year4.json), so the same script
works for one year or for the final merged evaluation.

Check types (set per question in the JSON file):
  contains_all : Pass only if ALL keywords appear in the answer.
  refusal      : Pass only if the agent says it could not find the information.
  redirect     : Pass if the agent politely says it only answers from the documents.
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from rag import ask

BASE = Path(__file__).parent
RESULTS_MD = BASE / "evaluation_results.md"
RESULTS_JSON = BASE / "evaluation_results.json"
PAUSE_SECONDS = 1.0  # small pause between questions to stay under API rate limits


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes
    text = re.sub(r"[\u2010-\u2015]", "-", text)               # special hyphens
    text = re.sub(r"\s+", " ", text)                            # also fixes non-breaking spaces
    return text


def check(answer: str, rule: dict) -> bool:
    a = normalize(answer)
    kind = rule["type"]
    if kind == "refusal":
        return "couldn't find this information" in a
    if kind == "redirect":
        return any(normalize(k) in a for k in rule["keywords"])
    return all(normalize(k) in a for k in rule["keywords"])


def load_questions() -> list[dict]:
    questions = []
    for path in sorted(BASE.glob("questions*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            item["file"] = path.name
        questions.extend(items)
    return questions


def ask_with_retry(question: str, attempts: int = 3) -> str:
    for attempt in range(attempts):
        try:
            return ask(question)["answer"]
        except Exception as error:
            if attempt == attempts - 1:
                return f"[ERROR] {error}"
            time.sleep(5 * (attempt + 1))  # wait longer after each failure


def cell(text: str) -> str:
    """Make text safe for a markdown table cell."""
    return text.replace("|", "/").replace("\n", " ").strip()


def main():
    questions = load_questions()
    if not questions:
        print("No questions*.json files found.")
        return

    rows = []
    for number, q in enumerate(questions, 1):
        answer = ask_with_retry(q["question"])
        passed = check(answer, q["check"])
        rows.append({**q, "no": number, "actual": answer, "result": "Pass" if passed else "Fail"})
        print(f"[{number}/{len(questions)}] {rows[-1]['result']}  {q['question']}")
        time.sleep(PAUSE_SECONDS)

    total = len(rows)
    passed = sum(r["result"] == "Pass" for r in rows)
    score = 100 * passed / total

    per_category = defaultdict(lambda: [0, 0])
    for r in rows:
        per_category[r["category"]][1] += 1
        per_category[r["category"]][0] += r["result"] == "Pass"

    lines = ["# Evaluation Results", "",
             f"**Overall score: {passed}/{total} ({score:.1f}%)**", "",
             "| Category | Passed | Total |", "|---|---|---|"]
    for cat in sorted(per_category):
        p, t = per_category[cat]
        lines.append(f"| {cat} | {p} | {t} |")
    lines += ["", "| No. | Question | Expected Answer | Actual Agent Answer | Result |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['no']} | {cell(r['question'])} | {cell(r['expected'])} | "
                     f"{cell(r['actual'])} | {r['result']} |")

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nOverall score: {passed}/{total} ({score:.1f}%)")
    for cat in sorted(per_category):
        p, t = per_category[cat]
        print(f"  {cat}: {p}/{t}")
    print(f"\nSaved: {RESULTS_MD.name} and {RESULTS_JSON.name}")
    failed = [r for r in rows if r["result"] == "Fail"]
    if failed:
        print("Review the failed ones manually: the keyword check can be too strict.")


if __name__ == "__main__":
    main()
