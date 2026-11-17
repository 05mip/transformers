"""Hand review in File Explorer: export images to keep/reject folders, then import the folders back.

    python cs117_project/review_folders.py export
        -> data/review_view/<class>/keep/   (copies of images currently marked keep)
           data/review_view/<class>/reject/ (copies of images marked reject)
           data/review_view/<class>/unreviewed/
    Move files between the folders to change decisions. To ADD your own images (e.g. saved from a
    Google search), drop them into <class>/keep/; optionally list their source URLs in
    data/review_view/manual_sources.csv (columns: filename,url).

    python cs117_project/review_folders.py import
        -> rewrites data/review.csv from folder membership; new images are copied into data/raw/<class>/
           and added to manifest_raw.csv with source "manual".

Folder copies are git-ignored; review.csv and manifest_raw.csv are the committed record.
"""
import csv
import hashlib
import shutil
import sys

from PIL import Image

from collect_images import FIELDS, MANIFEST, THUMB_WIDTH
from common import CLASSES, DATA, RAW_DIR, REVIEW

VIEW = DATA / "review_view"
STATES = ("keep", "reject", "unreviewed")


def load_review():
    if not REVIEW.exists():
        return {}
    return {r["file"]: r["decision"] for r in csv.DictReader(REVIEW.open(newline="", encoding="utf-8"))}


def export():
    review = load_review()
    manifest = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if VIEW.exists():
        shutil.rmtree(VIEW)
    counts = {}
    for r in manifest:
        if r["label"] not in CLASSES or not (RAW_DIR / r["file"]).exists():
            continue
        state = review.get(r["file"], "unreviewed")
        d = VIEW / r["label"] / state
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW_DIR / r["file"], d / r["file"].split("/")[-1])
        counts[(r["label"], state)] = counts.get((r["label"], state), 0) + 1
    for c in CLASSES:
        for s in STATES:
            (VIEW / c / s).mkdir(parents=True, exist_ok=True)
        print(f"{c:9s} " + "  ".join(f"{s} {counts.get((c, s), 0):4d}" for s in STATES))
    print(f"exported to {VIEW}")


def import_():
    manifest = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    by_name = {r["file"].split("/")[-1]: r for r in manifest}
    sources = {}
    src_csv = VIEW / "manual_sources.csv"
    if src_csv.exists():
        sources = {r["filename"]: r["url"] for r in csv.DictReader(src_csv.open(newline="", encoding="utf-8"))}
    review, added = {}, 0
    for c in CLASSES:
        for state in STATES:
            for p in sorted((VIEW / c / state).glob("*")):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                row = by_name.get(p.name)
                if row is None:  # a new image the reviewer added by hand
                    content = p.read_bytes()
                    sha = hashlib.sha1(content).hexdigest()
                    fname = f"{c}/{c}_manual_{sha[:12]}.jpg"
                    img = Image.open(p).convert("RGB")
                    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
                    img.save(RAW_DIR / fname, "JPEG", quality=90)
                    row = dict(file=fname, label=c, source="manual", source_id=p.name, page_url="",
                               image_url=sources.get(p.name, ""), license="unknown (manual; evaluation only)",
                               author=sources.get(p.name, "").split("/")[2] if sources.get(p.name, "").count("/") >= 2
                               else f"manual:{p.name}", sha1=sha)
                    manifest.append(row)
                    by_name[fname.split("/")[-1]] = row
                    added += 1
                elif row["label"] != c:  # moved to another class's folder: relabel
                    row["label"] = c
                if state != "unreviewed":
                    review[row["file"]] = {"file": row["file"], "label": c, "decision": state}
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(manifest)
    old = load_review()
    rows = [{"file": f, "label": next((r["label"] for r in manifest if r["file"] == f), ""), "decision": d}
            for f, d in old.items() if f not in review and not any(r["file"] == f and r["label"] in CLASSES for r in manifest)]
    rows += list(review.values())  # keep decisions for dropped classes (e.g. folding) untouched
    with REVIEW.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "decision"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))
    for c in CLASSES:
        print(f"{c:9s} keep {sum(r['label'] == c and r['decision'] == 'keep' for r in review.values()):4d}")
    print(f"added {added} new manual images; review.csv updated")


if __name__ == "__main__":
    {"export": export, "import": import_}[sys.argv[1]]()
