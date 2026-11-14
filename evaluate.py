"""
evaluate.py
Runs the test questions through the RAG system, grades every answer
(Pass/Fail), and writes evaluation_results.json / evaluation_results.md.

Usage:
    python evaluate.py [--questions year2_questions.json] [--limit N] [--delay 5]
    python evaluate.py --retry-errors      # re-run only questions that errored

Grading rules (from the project guidelines):
  * Category F (missing information): PASS only if the agent gives the exact
    refusal sentence.
  * Everything else: an LLM judge compares the answer with the expected answer
    (correct and supported = PASS; incorrect, incomplete or invented = FAIL).
  * "Error" = the LLM/API failed (busy, wrong model name, bad key). It is NOT a
    Fail of the RAG; fix the cause and re-run with --retry-errors.
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import (REFUSAL, RETRY_ATTEMPTS, RETRY_BASE_WAIT, AcademicRAG,
                 _is_temporary_error, build_providers)

RESULTS_JSON = Path("evaluation_results.json")
RESULTS_MD = Path("evaluation_results.md")

JUDGE = """You are grading an academic information chatbot.

Rules:
- PASS only if the actual answer is correct, complete and consistent with the
  expected answer. Wording and the order of list items may differ.
- FAIL if it is incorrect, incomplete, invented, or contradicts the expected answer.
- For out-of-scope questions, PASS only if the bot politely declines or
  redirects and gives no opinion or recommendation.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Reply with exactly one word: PASS or FAIL."""


def _clean(s: str) -> str:
    return " ".join(s.replace("\u2019", "'").lower().split())


def build_judge():
    """Returns judge(inputs) -> text or None. Uses retry + provider fallback."""
    prompt = ChatPromptTemplate.from_template(JUDGE)
    chains = [(name, prompt | llm | StrOutputParser())
              for name, llm in build_providers()]

    def judge(inputs: dict):
        for name, chain in chains:
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    return chain.invoke(inputs)
                except Exception as e:
                    if _is_temporary_error(e) and attempt < RETRY_ATTEMPTS:
                        wait = RETRY_BASE_WAIT * attempt
                        print(f"  [judge: {name} busy, retrying in {wait}s...]")
                        time.sleep(wait)
                        continue
                    print(f"  [judge: {name} failed: {type(e).__name__}: {str(e)[:120]}]")
                    break
        return None

    return judge


def grade(item, actual, judge):
    """Returns 'Pass', 'Fail' or 'Error'."""
    if item["category"].strip().upper().startswith("F"):
        return "Pass" if _clean(REFUSAL) in _clean(actual) else "Fail"

    verdict = judge({"question": item["question"],
                     "expected": item["expected_answer"],
                     "actual": actual})
    if verdict is None:
        return "Error"
    verdict = verdict.strip().upper()
    if verdict.startswith("PASS"):
        return "Pass"
    if verdict.startswith("FAIL"):
        return "Fail"
    return "Error"  # judge answered something unexpected


def write_reports(results):
    total = len(results)
    passed = sum(r["result"] == "Pass" for r in results)
    errors = sum(r["result"] == "Error" for r in results)
    score = 100 * passed / total if total else 0.0

    cats = {}
    for r in results:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += r["result"] == "Pass"
        c[1] += 1

    RESULTS_JSON.write_text(
        json.dumps({"score_percent": round(score, 1), "passed": passed,
                    "errors": errors, "total": total, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    def cell(s):
        return str(s).replace("|", "\\|").replace("\n", " ")

    md = [f"# Evaluation results\n\n**Overall score: {passed}/{total} = {score:.1f}%**\n"]
    if errors:
        md.append(f"> {errors} question(s) ended with Error (LLM/API failure). "
                  "Re-run with `--retry-errors`.\n")
    md += ["| Category | Passed |", "|---|---|"]
    md += [f"| {c} | {p}/{t} |" for c, (p, t) in cats.items()]
    md += ["", "| No. | Question | Expected Answer | Actual Agent Answer | Result |",
           "|---|---|---|---|---|"]
    md += [f"| {r['id']} | {cell(r['question'])} | {cell(r['expected_answer'])} | "
           f"{cell(r['actual_answer'])} | {r['result']} |" for r in results]
    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"\nOverall score: {passed}/{total} = {score:.1f}%")
    if errors:
        print(f"WARNING: {errors} question(s) ended with Error. "
              "Fix the cause, then run: python evaluate.py --retry-errors")
    for c, (p, t) in cats.items():
        print(f"  {c:<30} {p}/{t}")
    print(f"Saved {RESULTS_JSON} and {RESULTS_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="year2_questions.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0,
                    help="seconds to wait between questions (use 5 on free API tiers)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run only the questions that errored in the last run")
    args = ap.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    previous = {}
    if args.retry_errors:
        if not RESULTS_JSON.exists():
            raise SystemExit("No evaluation_results.json found. Run without --retry-errors first.")
        old = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["results"]
        previous = {r["id"]: r for r in old}

    rag = AcademicRAG()
    judge = build_judge()

    results = []
    for it in items:
        old = previous.get(it["id"])
        if args.retry_errors and old and old["result"] != "Error":
            results.append(old)  # keep the earlier Pass/Fail
            continue

        out = rag.ask(it["question"])
        actual = out["answer"]
        res = grade(it, actual, judge) if out["ok"] else "Error"
        results.append({**it, "actual_answer": actual, "result": res})
        print(f"[{it['id']:>3}] {res:<5} {it['question']}")
        time.sleep(args.delay)

    write_reports(results)


if __name__ == "__main__":
    main()