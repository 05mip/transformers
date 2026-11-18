"""Build the fixed train-pool / test split from the cleaned web images (+ phone photos as test-only).

Steps:
  1. Use every image in data/raw/<class>/ (hand-reviewed against data/labeling_criteria.md).
     Optionally cap images per source (photographer / domain) at --max-per-source per class.
     Default 0 = no cap: the team decided on 2026-11-17 to use every hand-reviewed image.
  2. Perceptual-hash (64-bit dHash) every image; images within Hamming distance <= --dup-thresh are
     near-duplicates. Within a class, keep one image per near-duplicate group. Groups that span classes
     (same picture under two labels) are dropped entirely as ambiguous.
  3. Per class, shuffle with a fixed seed: the first --n-test go to `test`, the rest to `pool`
     (few-shot training examples are drawn only from `pool`).
  4. Phone photos in data/phone/<class>/ are added with split `phone` (never used for training).

Writes data/splits.csv (committed). Run once; the test split is then frozen.

Usage:
    python cs117_project/make_splits.py --n-test 20
"""
import argparse
import csv
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from common import CLASSES, MANIFEST, MANUAL_SOURCES, PHONE_DIR, RAW_DIR, SPLITS, image_path


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--dup-thresh", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-source", type=int, default=0, help="0 = no cap (team decision 2026-11-17)")
    args = ap.parse_args()

    # source of each image: manifest author (photographer / domain), else manual_sources.csv domain
    source = {r["file"].split("/")[-1]: r["author"].strip().lower() for r in
              csv.DictReader(MANIFEST.open(newline="", encoding="utf-8"))}
    if MANUAL_SOURCES.exists():
        for r in csv.DictReader(MANUAL_SOURCES.open(newline="", encoding="utf-8")):
            # author = photographer (Commons artist, "flickr:<user>", or institution). Blank -> the image
            # is its own source; the hosting domain is NOT used, since it lumps unrelated photographers.
            source.setdefault(r["filename"], (r.get("author") or "").strip().lower())
    kept = []
    for c in CLASSES:
        d = RAW_DIR / c
        kept += [(f"{c}/{p.name}", c) for p in sorted(d.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    random.Random(args.seed).shuffle(kept)  # which images survive the per-source cap is seeded, not by name
    per_source, items = defaultdict(int), []
    for f, c in kept:
        key = (c, source.get(f.split("/")[-1]) or f)  # unknown source -> its own source
        if not args.max_per_source or per_source[key] < args.max_per_source:
            per_source[key] += 1
            items.append((f, c))
    print(f"{len(kept)} images in data/raw, {len(items)} after per-source cap of {args.max_per_source}")

    hashes = [dhash(image_path(f)) for f, _ in items]
    parent = list(range(len(items)))  # union-find over near-duplicates

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= args.dup_thresh:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)

    per_class = defaultdict(list)
    n_cross, n_dup = 0, 0
    for members in groups.values():
        labels = {items[i][1] for i in members}
        if len(labels) > 1:
            n_cross += len(members)
            continue
        n_dup += len(members) - 1
        per_class[labels.pop()].append(items[min(members)][0])  # one representative per group
    print(f"removed {n_dup} within-class near-duplicates, {n_cross} images in cross-class duplicate groups")

    rng = random.Random(args.seed)
    rows = []
    for c in CLASSES:
        files = sorted(per_class[c])
        rng.shuffle(files)
        test, pool = files[:args.n_test], files[args.n_test:]
        rows += [(f, c, "test") for f in test] + [(f, c, "pool") for f in pool]
        print(f"  {c:9s} test {len(test):3d}  pool {len(pool):3d}")
        if len(test) < args.n_test:
            print(f"  WARNING: {c} has fewer than {args.n_test} test images")

    for c in CLASSES:
        d = PHONE_DIR / c
        if d.is_dir():
            phone = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            rows += [(f"phone/{c}/{p.name}", c, "phone") for p in phone]
            print(f"  {c:9s} phone {len(phone):3d}")

    with SPLITS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "split"])
        w.writerows(rows)
    print(f"wrote {SPLITS}")


if __name__ == "__main__":
    main()
