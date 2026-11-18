"""Few-shot sweep: logistic regression on frozen embeddings, k shots per class drawn from the pool.

For each model, k in --shots, and seed in --seeds: sample k pool images per class (seeded), fit a
logistic-regression classifier on their L2-normalized embeddings, evaluate top-1 on test (and phone).
Regularization C is chosen by cross-validation on the k-shot training set only (never on test);
for k = 1 there is nothing to cross-validate, so C is fixed at 10.
The 0-shot point for CLIP/SigLIP comes from the zero-shot prompt classifier (see baselines.py).

With --cv: for each of the K group-aware folds in data/folds.csv, test = that fold
and the k shots are drawn from the other folds. Reports mean +- std over K folds x seeds, per-fold
means, balanced accuracy, mean per-class accuracy, a pooled confusion matrix (first seed, summed over
folds, so each image is counted once) and per-fold model ranks.

Usage (after embed.py):
    python cs117_project/fewshot.py
    python cs117_project/fewshot.py --cv --shots 1 2 4 8 16 32
Output: results/fewshot_test.json (or results/fewshot_cv.json with --cv)
"""
import argparse
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from common import CLASSES, MODELS, RESULTS, TEXT_MODELS, read_folds, read_splits, run_info, write_json
from evaluation import balanced_acc, load_all_folds, load_split, summarize, text_feats

C_GRID = np.logspace(-2, 3, 11)


def fit(X, y, k):
    if k == 1:
        clf = LogisticRegression(C=10.0, max_iter=2000)
    else:
        clf = LogisticRegressionCV(Cs=C_GRID, cv=min(k, 5), max_iter=2000)
    return clf.fit(X, y)


def stats(xs):
    return {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4)}


def confusion(y, pred):
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    return {"rows_true_cols_pred": CLASSES, "matrix": conf.tolist()}


def run_cv(args):
    n_folds = max(r["fold"] for r in read_folds()) + 1
    out = {"run": run_info(), "classes": CLASSES, "metric": "top-1 accuracy (also balanced accuracy)",
           "protocol": {"folds": n_folds, "fold_file": "data/folds.csv (StratifiedGroupKFold, grouped by source)",
                        "shots": args.shots, "seeds": args.seeds, "classifier": "logistic regression",
                        "C_selection": "k-fold CV on the k-shot training set (k=1: C=10 fixed)",
                        "aggregate": f"mean +- std over {n_folds} folds x {len(args.seeds)} seeds"},
           "results": {}}
    fold_mean = {}  # (model, k) -> per-fold mean top-1 over seeds, for the ranking
    for m in args.models:
        X, y, fold = load_all_folds(m)
        res = {}
        if m in TEXT_MODELS:  # 0 shots = zero-shot prompt classifier, no training data
            T, _ = text_feats(m)
            pred = (X @ T.T).argmax(1)
            per_fold = [float((pred[fold == f] == y[fold == f]).mean()) for f in range(n_folds)]
            per_fold_bal = [balanced_acc(y[fold == f], pred[fold == f]) for f in range(n_folds)]
            res["0"] = {**stats(per_fold), "balanced": stats(per_fold_bal), "per_fold": [round(a, 4) for a in per_fold],
                        "pooled_top1": round(float((pred == y).mean()), 4),
                        "per_class_top1": {c: round(float((pred[y == i] == i).mean()), 4) for i, c in enumerate(CLASSES)},
                        "pooled_confusion": confusion(y, pred)}
            fold_mean[(m, 0)] = per_fold
            print(f"  {m} k= 0: top-1 {res['0']['mean']:.3f}+-{res['0']['std']:.3f} (over folds)  "
                  f"pooled {res['0']['pooled_top1']:.3f}")
        for k in args.shots:
            accs, bals, per_fold, per_class_runs, skipped = [], [], [], [], []
            pooled_pred = np.full(len(y), -1)
            for f in range(n_folds):
                tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                min_train = min(int((y[tr] == c).sum()) for c in range(len(CLASSES)))
                if k > min_train:
                    skipped.append(f)
                    continue
                f_accs = []
                for seed in args.seeds:
                    rng = np.random.default_rng(seed)
                    idx = np.concatenate([rng.choice(tr[y[tr] == c], k, replace=False) for c in range(len(CLASSES))])
                    pred = fit(X[idx], y[idx], k).predict(X[te])
                    f_accs.append(float((pred == y[te]).mean()))
                    bals.append(balanced_acc(y[te], pred))
                    per_class_runs.append([float((pred[y[te] == c] == c).mean()) for c in range(len(CLASSES))])
                    if seed == args.seeds[0]:
                        pooled_pred[te] = pred
                accs += f_accs
                per_fold.append(float(np.mean(f_accs)))
            if skipped:
                print(f"  {m}: skip k={k} (folds {skipped} have a class with fewer than {k} training images)")
                continue
            res[str(k)] = {**stats(accs), "balanced": stats(bals), "per_fold": [round(a, 4) for a in per_fold],
                           "per_run": [round(a, 4) for a in accs],
                           "per_class_top1": {c: round(float(v), 4) for c, v in zip(CLASSES, np.mean(per_class_runs, 0))},
                           "pooled_confusion_seed0": confusion(y, pooled_pred)}
            fold_mean[(m, k)] = per_fold
            print(f"  {m} k={k:2d}: top-1 {res[str(k)]['mean']:.3f}+-{res[str(k)]['std']:.3f}  "
                  f"balanced {res[str(k)]['balanced']['mean']:.3f}  per fold {[round(a, 3) for a in per_fold]}")
        out["results"][m] = {"cv": res}

    # Ranking stability: in how many folds does each model rank first at each k?
    ranking = {}
    for k in [0] + list(args.shots):
        ms = [m for m in args.models if (m, k) in fold_mean]
        if len(ms) < 2:
            continue
        firsts = {m: 0 for m in ms}
        orders = []
        for f in range(n_folds):
            order = sorted(ms, key=lambda m: -fold_mean[(m, k)][f])
            firsts[order[0]] += 1
            orders.append(order)
        same = all(o == orders[0] for o in orders)
        ranking[str(k)] = {"folds_ranked_first": firsts, "order_per_fold": orders, "same_order_all_folds": same}
        print(f"  ranking k={k:2d}: folds ranked first {firsts}; same full order in every fold: {same}")
    out["ranking"] = ranking
    write_json(RESULTS / "fewshot_cv.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cv", action="store_true", help="group-aware K-fold CV over data/folds.csv")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if args.cv:
        return run_cv(args)

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
