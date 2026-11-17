"""Shared evaluation helpers: load cached features for a split, accuracy with bootstrap CI, confusion."""
import numpy as np

from common import CLASSES, FEATURES, read_splits


def load_split(model, split):
    """Return (X, y, files) for one split, aligned via the cached file list. y indexes common.CLASSES."""
    cache = np.load(FEATURES / f"{model}.npz")
    index = {f: i for i, f in enumerate(cache["files"])}
    rows = read_splits(split)
    missing = [r["file"] for r in rows if r["file"] not in index]
    if missing:
        raise SystemExit(f"{len(missing)} {split} files not in features/{model}.npz; rerun embed.py")
    X = cache["feats"][[index[r["file"]] for r in rows]]
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    return X, y, [r["file"] for r in rows]


def text_feats(model):
    cache = np.load(FEATURES / f"{model}.npz")
    return cache["text_feats"], [str(p) for p in cache["prompts"]]


def summarize(y_true, y_pred, n_boot=1000, seed=0):
    """Top-1 accuracy, 95% bootstrap CI over test images, per-class accuracy, confusion matrix."""
    correct = (y_true == y_pred).astype(float)
    rng = np.random.default_rng(seed)
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)]
    present = sorted(set(y_true.tolist()))
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    return {
        "top1": round(float(correct.mean()), 4),
        "top1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "n_images": int(len(y_true)),
        "per_class_top1": {CLASSES[c]: round(float(correct[y_true == c].mean()), 4) for c in present},
        "per_class_n": {CLASSES[c]: int((y_true == c).sum()) for c in present},
        "confusion": {"rows_true_cols_pred": CLASSES, "matrix": conf.tolist()},
    }
