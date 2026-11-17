"""Few-shot sweep: logistic regression on frozen embeddings, k shots per class drawn from the pool.

For each model, k in --shots, and seed in --seeds: sample k pool images per class (seeded), fit a
logistic-regression classifier on their L2-normalized embeddings, evaluate top-1 on test (and phone).
Regularization C is chosen by cross-validation on the k-shot training set only (never on test);
for k = 1 there is nothing to cross-validate, so C is fixed at 10.
The 0-shot point for CLIP/SigLIP comes from the zero-shot prompt classifier (see baselines.py).

Usage (after embed.py):
    python cs117_project/fewshot.py
Output: results/fewshot_test.json
"""
import argparse
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from common import CLASSES, MODELS, RESULTS, TEXT_MODELS, read_splits, run_info, write_json
from evaluation import load_split, summarize, text_feats

C_GRID = np.logspace(-2, 3, 11)


def fit(X, y, k):
    if k == 1:
        clf = LogisticRegression(C=10.0, max_iter=2000)
    else:
        clf = LogisticRegressionCV(Cs=C_GRID, cv=min(k, 5), max_iter=2000)
    return clf.fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    eval_splits = [s for s in ("test", "phone") if read_splits(s)]
    out = {"run": run_info(), "classes": CLASSES, "metric": "top-1 accuracy",
           "protocol": {"shots": args.shots, "seeds": args.seeds, "classifier": "logistic regression",
                        "C_selection": "k-fold CV on the k-shot training set (k=1: C=10 fixed)"},
           "results": {}}
    for m in args.models:
        Xp, yp, _ = load_split(m, "pool")
        evals = {s: load_split(m, s)[:2] for s in eval_splits}
        min_pool = min(int((yp == c).sum()) for c in range(len(CLASSES)))
        res = {s: {} for s in eval_splits}
        if m in TEXT_MODELS:
            T, _ = text_feats(m)
            for s, (X, y) in evals.items():
                res[s]["0"] = {"mean": summarize(y, (X @ T.T).argmax(1))["top1"], "std": 0.0, "per_seed": None}
        for k in args.shots:
            if k > min_pool:
                print(f"  {m}: skip k={k} (smallest class pool has {min_pool})")
                continue
            accs = {s: [] for s in eval_splits}
            last = {}
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                idx = np.concatenate([rng.choice(np.flatnonzero(yp == c), k, replace=False) for c in range(len(CLASSES))])
                clf = fit(Xp[idx], yp[idx], k)
                for s, (X, y) in evals.items():
                    pred = clf.predict(X)
                    accs[s].append(float((pred == y).mean()))
                    last[s] = summarize(y, pred, n_boot=1)
            for s in eval_splits:
                res[s][str(k)] = {"mean": round(float(np.mean(accs[s])), 4), "std": round(float(np.std(accs[s])), 4),
                                  "per_seed": [round(a, 4) for a in accs[s]]}
                if k == max(args.shots):
                    res[s][str(k)]["confusion_last_seed"] = last[s]["confusion"]
            print(f"  {m} k={k:2d}: " + "  ".join(f"{s} {res[s][str(k)]['mean']:.3f}±{res[s][str(k)]['std']:.3f}"
                                                  for s in eval_splits))
        out["results"][m] = res
    write_json(RESULTS / "fewshot_test.json", out)


if __name__ == "__main__":
    main()
