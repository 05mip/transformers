"""Make numbered contact sheets of the raw images, for hand-cleaning.

Each sheet is a grid of thumbnails with an index number; the matching sheet_index.csv maps
(sheet, number) -> file. Record decisions with record_review.py.

Usage:
    python cs117_project/contact_sheets.py [--classes hybrid cruiser]
Output: data/sheets/<class>_<n>.jpg and data/sheets/sheet_index.csv (git-ignored)
"""
import argparse
import csv

from PIL import Image, ImageDraw, ImageFont

from common import CLASSES, DATA, RAW_DIR, REVIEW

THUMB, COLS, ROWS = 160, 8, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--unreviewed", action="store_true", help="only images not yet in data/review.csv")
    args = ap.parse_args()
    reviewed = set()
    if args.unreviewed and REVIEW.exists():
        reviewed = {r["file"] for r in csv.DictReader(REVIEW.open(newline="", encoding="utf-8"))}
    out = DATA / "sheets"
    out.mkdir(exist_ok=True)
    manifest = list(csv.DictReader((DATA / "manifest_raw.csv").open(newline="", encoding="utf-8")))
    index = []
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for c in args.classes:
        files = [r["file"] for r in manifest if r["label"] == c and r["file"] not in reviewed]
        per = COLS * ROWS
        for s in range(0, len(files), per):
            sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "white")
            draw = ImageDraw.Draw(sheet)
            for i, f in enumerate(files[s:s + per]):
                im = Image.open(RAW_DIR / f).convert("RGB")
                im.thumbnail((THUMB - 4, THUMB - 4))
                x, y = (i % COLS) * THUMB, (i // COLS) * THUMB
                sheet.paste(im, (x + 2, y + 2))
                n = s + i
                draw.rectangle([x + 2, y + 2, x + 44, y + 26], fill="black")
                draw.text((x + 5, y + 3), str(n), fill="yellow", font=font)
                index.append((f"{c}_{s // per}", n, f))
            sheet.save(out / f"{c}_{s // per}.jpg", quality=85)
        print(f"{c}: {len(files)} images, {-(-len(files) // per)} sheets")
    with (out / "sheet_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet", "number", "file"])
        w.writerows(index)


if __name__ == "__main__":
    main()
