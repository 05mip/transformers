"""Record hand-review decisions from a contact-sheet pass into data/review.csv.

Every file shown on the given class's contact sheets (data/sheets/sheet_index.csv) is marked reviewed:
the listed numbers are `keep`, all others `reject` (fails data/labeling_criteria.md). Only `keep`
images are ever used by make_splits.py; unreviewed images are never used.

Usage:
    python cs117_project/record_review.py road "1,2,3,10-12"
"""
import csv
import sys

from common import DATA, REVIEW


def parse(spec):
    out = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    label, keep = sys.argv[1], parse(sys.argv[2])
    index = [r for r in csv.DictReader((DATA / "sheets" / "sheet_index.csv").open(encoding="utf-8"))
             if r["sheet"].rsplit("_", 1)[0] == label]
    shown = {int(r["number"]): r["file"] for r in index}
    bad = keep - shown.keys()
    if bad:
        raise SystemExit(f"numbers not on {label} sheets: {sorted(bad)}")
    rows = {}
    if REVIEW.exists():
        rows = {r["file"]: r for r in csv.DictReader(REVIEW.open(newline="", encoding="utf-8"))}
    for n, f in shown.items():
        rows[f] = {"file": f, "label": label, "decision": "keep" if n in keep else "reject"}
    with REVIEW.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "decision"])
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["file"]))
    n_keep = sum(r["label"] == label and r["decision"] == "keep" for r in rows.values())
    print(f"{label}: marked {len(shown)} reviewed ({len(keep)} keep this pass); total keep for class = {n_keep}")


if __name__ == "__main__":
    main()
