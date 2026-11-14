import json
from collections import Counter
from pathlib import Path

# add 4: "questions/questions_year4.json" when Year 4 arrives
FILES = {
    1: "questions/questions_year1.json",
    2: "questions/year2_questions.json",
    3: "questions/year3_questions.json",
}

all_q = []
for year, path in FILES.items():
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, it in enumerate(items, start=1):
        # Year 1 stores the answer under "expected"; the others use "expected_answer"
        expected = it.get("expected_answer") or it.get("expected")
        if not it.get("category") or not it.get("question") or not expected:
            raise SystemExit(f"Year {year}, question {i}: missing category/question/answer")
        all_q.append({
            "id": year * 100 + i,          # year1 -> 101.., year2 -> 201.., year3 -> 301..
            "category": it["category"],
            "question": it["question"],
            "expected_answer": expected,
        })
    print(f"Year {year}: {len(items)} questions")
    for cat, n in sorted(Counter(it["category"] for it in items).items()):
        print(f"   {cat}: {n}")

Path("all_questions.json").write_text(
    json.dumps(all_q, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{len(all_q)} questions saved to all_questions.json")