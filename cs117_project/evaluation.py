"""Shared evaluation helpers: load cached features for a split or CV fold, accuracy with bootstrap CI,
balanced accuracy, confusion."""
import numpy as np

from common import CLASSES, FEATURES, read_folds, read_splits


def _features(model, rows, what):
    cache = np.load(FEATURES / f"{model}.npz")
    index = {f: i for i, f in enumerate(cache["files"])}
    missing = [r["file"] for r in rows if r["file"] not in index]
    if missing:
        raise SystemExit(f"{len(missing)} {what} files not in features/{model}.npz; rerun embed.py")
    X = cache["feats"][[index[r["file"]] for r in rows]]
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    return X, y, [r["file"] for r in rows]


def load_split(model, split):
    """Return (X, y, files) for one split, aligned via the cached file list. y indexes common.CLASSES."""
    return _features(model, read_splits(split), split)


def load_fold(model, fold):
    """CV fold `fold` of data/folds.csv: ((X_train, y_train), (X_test, y_test)). Train = all other folds."""
    rows = read_folds()
    Xtr, ytr, _ = _features(model, [r for r in rows if r["fold"] != fold], f"fold!={fold}")
    Xte, yte, _ = _features(model, [r for r in rows if r["fold"] == fold], f"fold={fold}")
    return (Xtr, ytr), (Xte, yte)


def load_all_folds(model):
    """Every image in data/folds.csv: (X, y, fold) arrays."""
    rows = read_folds()
    X, y, _ = _features(model, rows, "folds.csv")
    return X, y, np.array([r["fold"] for r in rows])


def balanced_acc(y_true, y_pred):
    """Mean of per-class accuracy over the classes present in y_true (robust to unbalanced test sets)."""
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def text_feats(model):
    cache = np.load(FEATURES / f"{model}.npz")
    return cache["text_feats"], [str(p) for p in cache["prompts"]]


def summarize(y_true, y_pred, n_boot=1000, seed=0):
    """Top-1 accuracy, 95% bootstrap CI over test images, balanced accuracy, per-class accuracy, confusion."""
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
        "balanced_acc": round(balanced_acc(y_true, y_pred), 4),
        "n_images": int(len(y_true)),
        "per_class_top1": {CLASSES[c]: round(float(correct[y_true == c].mean()), 4) for c in present},
        "per_class_n": {CLASSES[c]: int((y_true == c).sum()) for c in present},
        "confusion": {"rows_true_cols_pred": CLASSES, "matrix": conf.tolist()},
    }
