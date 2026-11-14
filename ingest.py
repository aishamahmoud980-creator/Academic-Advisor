"""
ingest.py
Loads every TXT file under data/ (all years), parses course and semester
files, runs consistency checks, and returns LangChain Documents with metadata.

Works the same for one year or all four: it scans data/year*/ automatically.
Run `python ingest.py` to preview what will be indexed and see any warnings.
"""
import re
from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent / "data"

COURSE_FIELDS = [
    "Course Code", "Course Name", "Department", "Year", "Term",
    "Credit Hours", "Prerequisites", "Unlocks", "Description",
]
_FIELD_RE = re.compile(r"^(" + "|".join(COURSE_FIELDS) + r"):\s*(.*)$")
CODE_PAT = r"[A-Z]{2,4}\d{3,4}"      # CCS2201, CS401, GM315, IS371 ...
_SEM_COURSE_RE = re.compile(
    r"^-\s*(" + CODE_PAT + r"):\s*(.+?)\s*\((\d+(?:\.\d+)?)\s*Credit Hours?\)\s*$"
)


def _int(value, default=-1):
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _num(value):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _fmt(x):
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else str(x)


def parse_course_file(path: Path) -> dict:
    fields, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _FIELD_RE.match(line)
        if m:
            current = m.group(1)
            fields[current] = m.group(2).strip()
        elif current and line.strip():  # continuation of a long field
            fields[current] += " " + line.strip()
    return fields


def _terms(value):
    """'7' -> [7];  '7 and 8' -> [7, 8];  empty/invalid -> [-1]"""
    nums = [int(n) for n in re.findall(r"\d+", str(value))]
    return nums or [-1]


def parse_semester_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    year = re.search(r"^Year:\s*(\d+)", text, re.M)
    term = re.search(r"^Term:\s*(\d+)", text, re.M)
    total = re.search(r"^Total Credit Hours:\s*([\d.]+)", text, re.M)
    courses, in_elective = [], False
    for raw in text.splitlines():
        line = raw.strip()
        m = _SEM_COURSE_RE.match(line)
        if m:
            courses.append({"code": m.group(1), "name": m.group(2),
                            "credits": float(m.group(3)), "elective": in_elective})
        elif line.lower().startswith(("courses", "total credit")):
            in_elective = False
        elif line and not line.startswith("-") and "elective" in line.lower():
            in_elective = True        # e.g. "Major Elective options (...)"
    core = {c["code"] for c in courses if not c["elective"]}
    for c in courses:                 # a course listed as required AND as option stays required
        if c["code"] in core:
            c["elective"] = False
    return {
        "year": _int(year.group(1)) if year else -1,
        "term": _int(term.group(1)) if term else -1,
        "total": float(total.group(1)) if total else None,
        "courses": courses,
        "text": text.strip(),
    }


def load_documents(data_dir: Path = DATA_DIR, verbose: bool = True):
    warns, docs = [], []
    courses = {}        # code -> metadata
    course_terms = {}   # code -> [terms]  (an elective may be offered in several terms)
    semesters = {}      # (year, term) -> parsed

    for year_dir in sorted(Path(data_dir).glob("year*")):
        if not year_dir.is_dir():
            continue
        folder_year = _int(re.sub(r"\D", "", year_dir.name))

        # ---- course files
        for path in sorted((year_dir / "courses").glob("*.txt")):
            f = parse_course_file(path)
            code = f.get("Course Code", "").strip().upper()
            rel = str(path.relative_to(data_dir))
            if not code:
                warns.append(f"{rel}: missing Course Code"); continue
            if path.stem.upper() != code:
                warns.append(f"{rel}: filename does not match Course Code {code}")
            for name in COURSE_FIELDS:
                if not f.get(name):
                    warns.append(f"{rel}: empty field '{name}'")
            year = _int(f.get("Year"))
            terms = _terms(f.get("Term"))
            if year != folder_year:
                warns.append(f"{rel}: Year {year} does not match folder {year_dir.name}")
            if terms == [-1]:
                warns.append(f"{rel}: Term '{f.get('Term', '')}' is not a number")
            meta = {
                "doc_id": f"course::{code}", "doc_type": "course",
                "course_code": code, "course_name": f.get("Course Name", ""),
                "department": f.get("Department", ""), "year": year, "term": terms[0],
                "terms": ",".join(map(str, terms)), "term_label": f.get("Term", ""),
                "credit_hours": _num(f.get("Credit Hours")) or 0.0,
                "prerequisites": f.get("Prerequisites", ""), "source": rel,
            }
            courses[code] = meta
            course_terms[code] = terms
            text = path.read_text(encoding="utf-8").strip()
            docs.append(Document(page_content=text, metadata=meta))

        # ---- semester files
        for path in sorted(year_dir.glob("term*.txt")):
            s = parse_semester_file(path)
            rel = str(path.relative_to(data_dir))
            if s["year"] != folder_year:
                warns.append(f"{rel}: Year {s['year']} does not match folder {year_dir.name}")
            semesters[(s["year"], s["term"])] = s
            docs.append(Document(page_content=s["text"], metadata={
                "doc_id": f"semester::y{s['year']}t{s['term']}",
                "doc_type": "semester", "year": s["year"], "term": s["term"],
                "source": rel,
            }))

    # ---- cross-checks between semester files and course files
    in_semester = set()
    elective_codes = {}   # (year, term) -> codes that are elective options
    for (y, t), s in semesters.items():
        seen, core_sum = set(), 0.0
        elective_codes[(y, t)] = {c["code"] for c in s["courses"] if c["elective"]}
        for c in s["courses"]:
            if c["code"] in seen:
                continue
            seen.add(c["code"])
            if not c["elective"]:
                core_sum += c["credits"]
            in_semester.add(c["code"])
            cm = courses.get(c["code"])
            if not cm:
                warns.append(f"Year {y} Term {t}: {c['code']} has no course file")
                continue
            if t not in course_terms[c["code"]]:
                warns.append(f"{c['code']}: course file Term {cm['term_label']} does not include semester file Term {t}")
            if cm["course_name"] != c["name"]:
                warns.append(f"{c['code']}: name differs between course and semester file")
            if abs(cm["credit_hours"] - c["credits"]) > 1e-6:
                warns.append(f"{c['code']}: credit hours differ between course and semester file")
        if s["total"] is None or abs(core_sum - s["total"]) > 1e-6:
            warns.append(f"Year {y} Term {t}: Total Credit Hours {s['total']} != sum of required courses {core_sum:g}")
    for code in courses:
        if code not in in_semester:
            warns.append(f"{code}: not listed in any semester file")

    # ---- derived summary documents (built from the files, nothing invented)
    # They let a top-4 retrieval answer "list everything in Term X / Year Y".
    by_term = {}
    for code, m in courses.items():
        for t in course_terms[code]:
            by_term.setdefault((m["year"], t), []).append(m)

    def prereq_locations(m):
        found = []
        for c in dict.fromkeys(re.findall(CODE_PAT, m["prerequisites"])):
            if c in courses:
                found.append(f"{c} (Year {courses[c]['year']}, Term {courses[c]['term_label']})")
        return f" | Prerequisite location: {'; '.join(found)}" if found else ""

    def line(m, with_term=False, elective=False):
        extra = f" | Term: {m['term_label']}" if with_term else ""
        tag = " | Elective option" if elective else ""
        return (f"- {m['course_code']}: {m['course_name']}{extra} | "
                f"Credit Hours: {_fmt(m['credit_hours'])} | Prerequisites: {m['prerequisites']}"
                f"{prereq_locations(m)}{tag}")

    term_totals = {}
    for (y, t), ms in sorted(by_term.items()):
        ms = sorted(ms, key=lambda m: m["course_code"])
        sem = semesters.get((y, t))
        elec = elective_codes.get((y, t), set())
        req = [m for m in ms if m["course_code"] not in elec]
        opt = [m for m in ms if m["course_code"] in elec]
        total = sem["total"] if sem and sem["total"] is not None else sum(m["credit_hours"] for m in req)
        term_totals[(y, t)] = total
        if not opt:
            text = (f"Term {t} (Year {y}) summary\n"
                    f"Courses offered in Term {t} of Year {y} with credit hours and prerequisites:\n"
                    + "\n".join(line(m) for m in ms)
                    + f"\nNumber of courses in Term {t}: {len(ms)}"
                    + f"\nTotal Credit Hours in Term {t}: {_fmt(total)}")
        else:
            text = (f"Term {t} (Year {y}) summary\n"
                    f"Required courses in Term {t} of Year {y} with credit hours and prerequisites:\n"
                    + "\n".join(line(m) for m in req)
                    + f"\nElective options available in Term {t} (students choose from these):\n"
                    + "\n".join(line(m, elective=True) for m in opt)
                    + f"\nNumber of required courses in Term {t}: {len(req)}"
                    + f"\nNumber of elective options in Term {t}: {len(opt)}"
                    + f"\nTotal Credit Hours in Term {t} (required courses): {_fmt(total)}")
        docs.append(Document(page_content=text, metadata={
            "doc_id": f"term_summary::y{y}t{t}", "doc_type": "term_summary",
            "year": y, "term": t, "source": "derived"}))

    for y in sorted({k[0] for k in by_term}):
        ms = sorted((m for m in courses.values() if m["year"] == y),
                    key=lambda m: (m["term"], m["course_code"]))
        y_terms = sorted(t for (yy, t) in by_term if yy == y)
        elec = set().union(*(elective_codes.get((y, t), set()) for t in y_terms))
        total = sum(term_totals[(y, t)] for t in y_terms)
        n_opt = sum(1 for m in ms if m["course_code"] in elec)
        count = (f"\nNumber of courses in Year {y}: {len(ms)}" if not n_opt else
                 f"\nNumber of required courses in Year {y}: {len(ms) - n_opt}"
                 f"\nNumber of elective options in Year {y}: {n_opt}")
        text = (f"Year {y} summary (Terms {', '.join(map(str, y_terms))})\n"
                f"All courses in Year {y} with credit hours and prerequisites:\n"
                + "\n".join(line(m, with_term=True, elective=m["course_code"] in elec) for m in ms)
                + count
                + f"\nTotal Credit Hours in Year {y}"
                + (" (required courses)" if n_opt else "") + f": {_fmt(total)}")
        docs.append(Document(page_content=text, metadata={
            "doc_id": f"year_summary::y{y}", "doc_type": "year_summary",
            "year": y, "source": "derived"}))

    if verbose:
        kinds = {}
        for d in docs:
            kinds[d.metadata["doc_type"]] = kinds.get(d.metadata["doc_type"], 0) + 1
        print(f"Loaded {len(docs)} documents: {kinds}")
        if warns:
            print(f"\n{len(warns)} warning(s):")
            for w in warns:
                print("  -", w)
        else:
            print("No consistency warnings.")
    return docs


if __name__ == "__main__":
    load_documents()