"""Build the fixed train-pool / test split from the cleaned web images (+ phone photos as test-only).

Steps:
  1. Keep only files marked `keep` in data/review.csv (hand review against data/labeling_criteria.md);
     unreviewed files are never used. Then cap images per source (Commons/Openverse photographer, or
     website domain for web-search images) at --max-per-source per class, so one photo session or
     one shop cannot dominate a class (and leak between test and pool).
  2. Perceptual-hash (64-bit dHash) every image; images within Hamming distance <= --dup-thresh are
     near-duplicates. Within a class, keep one image per near-duplicate group. Groups that span classes
     (same picture under two labels) are dropped entirely as ambiguous.
  3. Per class, shuffle with a fixed seed: the first --n-test go to `test`, the rest to `pool`
     (few-shot training examples are drawn only from `pool`).
  4. Phone photos in data/phone/<class>/ are added with split `phone` (never used for training).

Writes data/splits.csv (committed). Run once; the test split is then frozen.

Usage:
    python cs117_project/make_splits.py --n-test 40
"""
import argparse
import csv
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from common import CLASSES, DATA, PHONE_DIR, RAW_DIR, REVIEW, SPLITS


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--dup-thresh", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-source", type=int, default=8)
    args = ap.parse_args()

    keep = {r["file"] for r in csv.DictReader(REVIEW.open(newline="", encoding="utf-8")) if r["decision"] == "keep"}
    manifest = list(csv.DictReader((DATA / "manifest_raw.csv").open(newline="", encoding="utf-8")))
    kept = sorted((r for r in manifest if r["file"] in keep and r["label"] in CLASSES and (RAW_DIR / r["file"]).exists()),
                  key=lambda r: r["file"])
    random.Random(args.seed).shuffle(kept)  # which images survive the per-source cap is seeded, not by name
    per_source, items = defaultdict(int), []
    for r in kept:
        key = (r["label"], r["author"].strip().lower() or r["file"])  # no author -> its own source
        if per_source[key] < args.max_per_source:
            per_source[key] += 1
            items.append((r["file"], r["label"]))
    print(f"{len(manifest)} downloaded, {len(keep)} kept in hand review, "
          f"{len(items)} after per-source cap of {args.max_per_source}")

    hashes = [dhash(RAW_DIR / f) for f, _ in items]
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
