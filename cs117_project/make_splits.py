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

With --folds K, steps 3-4 are replaced by group-aware K-fold assignment: every
kept image goes to exactly one fold, stratified by class and grouped by source (photographer), so a
photo session never spans a test fold and its training folds. Writes data/folds.csv
(file,label,source,fold) and leaves data/splits.csv untouched.

Usage:
    python cs117_project/make_splits.py --n-test 20
    python cs117_project/make_splits.py --folds 4 --max-per-source 15
"""
import argparse
import csv
import random
from collections import defaultdict

import numpy as np
from PIL import Image

from common import CLASSES, FOLDS, MANIFEST, MANUAL_SOURCES, PHONE_DIR, RAW_DIR, SPLITS, image_path


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def source_key(f, label, source):
    """Group key for an image: class + photographer. Unknown photographer -> the image is its own group."""
    return f"{label}|{source.get(f.split('/')[-1]) or f}"


def per_source_all(kept, source):
    counts = defaultdict(int)
    for f, c in kept:
        counts[(c, source.get(f.split("/")[-1]) or f)] += 1
    return counts


def write_folds(per_class, source, args):
    from sklearn.model_selection import StratifiedGroupKFold

    files = [f for c in CLASSES for f in sorted(per_class[c])]
    labels = [f.split("/")[0] for f in files]
    groups = [source_key(f, lab, source) for f, lab in zip(files, labels)]
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_of = {}
    for k, (_, test_idx) in enumerate(sgkf.split(files, labels, groups)):
        for i in test_idx:
            fold_of[files[i]] = k

    # checks: every image in exactly one fold; no source group spans two folds
    assert len(fold_of) == len(files) == len(set(files)), "an image is missing from the folds or repeated"
    group_folds = defaultdict(set)
    for f, g in zip(files, groups):
        group_folds[g].add(fold_of[f])
    split_groups = [g for g, fs in group_folds.items() if len(fs) > 1]
    assert not split_groups, f"source groups split across folds: {split_groups[:5]}"

    print(f"{len(files)} images, {len(group_folds)} source groups, {args.folds} folds (seed {args.seed})")
    print("  fold  " + " ".join(f"{c:>8s}" for c in CLASSES) + "   total")
    for k in range(args.folds):
        n = [sum(1 for f, lab in zip(files, labels) if fold_of[f] == k and lab == c) for c in CLASSES]
        print(f"  {k:4d}  " + " ".join(f"{x:8d}" for x in n) + f"   {sum(n):5d}")
    n = [labels.count(c) for c in CLASSES]
    print("  all   " + " ".join(f"{x:8d}" for x in n) + f"   {sum(n):5d}")

    with FOLDS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "source", "fold"])
        for f, lab, g in zip(files, labels, groups):
            w.writerow([f, lab, g.split("|", 1)[1], fold_of[f]])
    print(f"wrote {FOLDS} (data/splits.csv left unchanged)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--dup-thresh", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-source", type=int, default=0, help="0 = no cap (team decision 2026-11-17)")
    ap.add_argument("--folds", type=int, default=0, help="K > 0: write data/folds.csv (K group-aware folds) "
                                                        "instead of data/splits.csv")
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
    if args.max_per_source:
        dropped = defaultdict(int)
        for (c, src), n in per_source_all(kept, source).items():
            if n > args.max_per_source:
                dropped[(c, src)] = n - args.max_per_source
        for (c, src), n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  cap dropped {n:3d} {c} images from source {src!r}")

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

    if args.folds:
        write_folds(per_class, source, args)
        return

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
