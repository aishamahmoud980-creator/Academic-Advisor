"""
check_data.py
Checks one year folder against the project rules BEFORE merging.

Usage (from the project folder):
    python check_data.py data/year3
"""
import re
import sys
from pathlib import Path

FIELDS = ["Course Code", "Course Name", "Department", "Year", "Term",
          "Credit Hours", "Prerequisites", "Unlocks", "Description"]
CODE = r"[A-Z]{3}\d{4}"

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def read(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        err(f"{path.name}: not UTF-8 encoded")
        return ""


def parse_course(path: Path):
    lines = read(path).splitlines()
    fields, order = {}, []
    for line in lines:
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m and m.group(1).strip() in FIELDS and m.group(1).strip() not in fields:
            name = m.group(1).strip()
            fields[name] = m.group(2).strip()
            order.append(name)
        elif order and line.strip():
            fields[order[-1]] += " " + line.strip()
    if order != FIELDS:
        missing = [f for f in FIELDS if f not in fields]
        err(f"{path.name}: fields wrong or out of order "
            f"(missing: {missing or 'none'}, found order: {order})")
    for f, v in fields.items():
        if not v:
            err(f"{path.name}: field '{f}' is empty "
                "(use 'Not specified in the provided source.' or 'None')")
    return fields


def parse_term(path: Path):
    text = read(path)
    year = re.search(r"^Year:\s*(\d+)", text, re.M)
    term = re.search(r"^Term:\s*(\d+)", text, re.M)
    total = re.search(r"^Total Credit Hours:\s*(\d+)", text, re.M)
    courses = re.findall(
        rf"^-\s*({CODE}):\s*(.+?)\s*\((\d+)\s*Credit Hours?\)\s*$", text, re.M)
    if not (year and term and total):
        err(f"{path.name}: missing Year / Term / Total Credit Hours line")
        return None
    if not courses:
        err(f"{path.name}: no courses found (check the '- CODE: Name (N Credit Hours)' format)")
    return {"year": int(year.group(1)), "term": int(term.group(1)),
            "total": int(total.group(1)),
            "courses": {c: (n, int(h)) for c, n, h in courses},
            "count_lines": len(courses)}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python check_data.py data/year3")
    folder = Path(sys.argv[1])
    ym = re.search(r"year(\d)", folder.name.lower())
    if not folder.is_dir() or not ym:
        raise SystemExit("Give the path of a year folder, e.g. data/year3")
    year = int(ym.group(1))
    dept = "Common" if year in (1, 2) else "Computer Science"
    expected_terms = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8)}[year]

    course_dir = folder / "courses"
    if not course_dir.is_dir():
        raise SystemExit(f"Missing folder: {course_dir}")

    course_files = {p.stem: p for p in course_dir.glob("*.txt")}
    listed = {}

    for t in expected_terms:
        tf = folder / f"term{t}.txt"
        if not tf.exists():
            err(f"Missing file: {tf.name}")
            continue
        info = parse_term(tf)
        if not info:
            continue
        if info["year"] != year:
            err(f"{tf.name}: says Year {info['year']}, expected {year}")
        if info["term"] != t:
            err(f"{tf.name}: says Term {info['term']}, expected {t}")
        s = sum(h for _, h in info["courses"].values())
        if s != info["total"]:
            err(f"{tf.name}: Total Credit Hours is {info['total']} but courses add up to {s}")
        if info["count_lines"] != len(info["courses"]):
            err(f"{tf.name}: a course code is listed twice")
        print(f"term{t}.txt: {len(info['courses'])} courses, {info['total']} credit hours")
        for code, (name, hours) in info["courses"].items():
            listed[code] = (t, name, hours)

    for code, (t, name, hours) in listed.items():
        path = course_files.get(code)
        if not path:
            err(f"{code}: listed in term{t}.txt but {code}.txt is missing in courses/")
            continue
        f = parse_course(path)
        if not f:
            continue
        if f.get("Course Code") != code:
            err(f"{path.name}: Course Code is '{f.get('Course Code')}', must match the filename")
        if f.get("Course Name") != name:
            err(f"{path.name}: name '{f.get('Course Name')}' differs from term file '{name}'")
        if f.get("Department") != dept:
            err(f"{path.name}: Department is '{f.get('Department')}', expected '{dept}'")
        if f.get("Year") != str(year):
            err(f"{path.name}: Year is '{f.get('Year')}', expected {year}")
        if f.get("Term") != str(t):
            err(f"{path.name}: Term is '{f.get('Term')}', expected {t}")
        if f.get("Credit Hours") != str(hours):
            err(f"{path.name}: Credit Hours '{f.get('Credit Hours')}' differs from term file ({hours})")
        for field in ("Prerequisites", "Unlocks"):
            v = f.get(field, "")
            if v in ("None", "Not specified in the provided source."):
                continue
            leftovers = re.sub(CODE, "", v)
            leftovers = re.sub(r"\b(AND|OR)\b|[(),\s]", "", leftovers)
            if leftovers:
                warnings.append(f"{path.name}: {field} has unusual text: '{v}'")

    for code in course_files:
        if code not in listed:
            err(f"{code}.txt exists in courses/ but is not listed in any term file")

    print(f"\n{len(listed)} courses listed, {len(course_files)} course files found")
    for w in warnings:
        print("WARNING:", w)
    if errors:
        print(f"\n{len(errors)} PROBLEM(S):")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("\nAll checks passed. This year folder is ready to merge.")


if __name__ == "__main__":
    main()