"""ingest.py - loads all course and semester TXT files from data/ and checks them."""
import re
from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent / "data"

COURSE_FIELDS = [
    "Course Code", "Course Name", "Department", "Year", "Term",
    "Credit Hours", "Prerequisites", "Unlocks", "Description",
]


def parse_course(text: str) -> dict:
    """Read 'Field: value' lines of a course file into a dict."""
    fields = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"^(" + "|".join(COURSE_FIELDS) + r"):\s*(.*)$", line)
        if match:
            current = match.group(1)
            fields[current] = match.group(2).strip()
        elif current == "Description" and line.strip():
            fields[current] += " " + line.strip()  # multi-line description
    return fields


def parse_term(text: str) -> dict:
    """Read a semester file: year, term, course list, total credit hours."""
    year = int(re.search(r"^Year:\s*(\d+)", text, re.M).group(1))
    term = int(re.search(r"^Term:\s*(\d+)", text, re.M).group(1))
    courses = re.findall(r"^-\s*(\S+):.*?\((\d+)\s*Credit Hours?\)", text, re.M)
    total = re.search(r"^Total Credit Hours:\s*(\d+)", text, re.M)
    return {
        "year": year,
        "term": term,
        "courses": [(code, int(cr)) for code, cr in courses],
        "total": int(total.group(1)) if total else None,
    }


def load_documents() -> list[Document]:
    """Return one Document per course file and one per semester file.

    The semester document also carries a short 'Course details' block built from
    the course files (prerequisites and unlocks), so questions about a whole term
    can be answered from a single retrieved chunk. The TXT files are not changed.
    """
    docs = []
    course_docs = []
    course_fields = {}

    for path in sorted(DATA_DIR.glob("year*/courses/*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        f = parse_course(text)
        course_fields[f["Course Code"]] = f
        course_docs.append(Document(
            page_content=text,
            metadata={
                "doc_type": "course",
                "year": int(f["Year"]),
                "term": int(f["Term"]),
                "course_code": f["Course Code"],
                "source": str(path.relative_to(DATA_DIR)),
            },
        ))

    for path in sorted(DATA_DIR.glob("year*/term*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        info = parse_term(text)
        details = []
        for code, _ in info["courses"]:
            f = course_fields.get(code)
            if f:
                details.append(f"- {code}: Prerequisites: {f['Prerequisites']}; Unlocks: {f['Unlocks']}")
        if details:
            text += "\n\nCourse details:\n" + "\n".join(details)
        docs.append(Document(
            page_content=text,
            metadata={
                "doc_type": "term",
                "year": info["year"],
                "term": info["term"],
                "source": str(path.relative_to(DATA_DIR)),
            },
        ))

    return docs + course_docs


def validate() -> list[str]:
    """Consistency checks from section 8 of the project guidelines."""
    problems = []
    courses = {}
    for path in DATA_DIR.glob("year*/courses/*.txt"):
        f = parse_course(path.read_text(encoding="utf-8"))
        missing = [k for k in COURSE_FIELDS if k not in f]
        if missing:
            problems.append(f"{path.name}: missing fields {missing}")
            continue
        if f["Course Code"] != path.stem:
            problems.append(f"{path.name}: Course Code does not match filename")
        courses[path.stem] = f

    for path in sorted(DATA_DIR.glob("year*/term*.txt")):
        info = parse_term(path.read_text(encoding="utf-8"))
        total = 0
        for code, credits in info["courses"]:
            total += credits
            if code not in courses:
                problems.append(f"{path.name}: {code} has no course file")
                continue
            c = courses[code]
            if int(c["Credit Hours"]) != credits:
                problems.append(f"{path.name}: {code} credit hours differ from course file")
            if int(c["Year"]) != info["year"] or int(c["Term"]) != info["term"]:
                problems.append(f"{path.name}: {code} year/term differ from course file")
        if info["total"] != total:
            problems.append(f"{path.name}: total is {info['total']} but courses sum to {total}")

    listed = {code for p in DATA_DIR.glob("year*/term*.txt")
              for code, _ in parse_term(p.read_text(encoding="utf-8"))["courses"]}
    for code in courses:
        if code not in listed:
            problems.append(f"{code}: course file is not listed in any term file")
    return problems


if __name__ == "__main__":
    documents = load_documents()
    n_courses = sum(d.metadata["doc_type"] == "course" for d in documents)
    n_terms = sum(d.metadata["doc_type"] == "term" for d in documents)
    print(f"Loaded {len(documents)} documents ({n_terms} term files, {n_courses} course files)")

    issues = validate()
    if issues:
        print(f"\n{len(issues)} problem(s) found:")
        for issue in issues:
            print(" -", issue)
    else:
        print("All consistency checks passed.")
