"""Baselines on the held-out splits: random guessing and zero-shot CLIP (plus zero-shot SigLIP).

Zero-shot = cosine similarity between the image embedding and the embedding of the fixed prompt
"a photo of a {class name}." (common.PROMPT_TEMPLATE); predict the most similar class. The prompt was
fixed before any scoring and is never tuned on test images.

With --cv: zero-shot needs no training, so it is scored once on every image in data/folds.csv (each
image tested once, bootstrap CI), and also per fold for the ranking comparison.

Usage (after embed.py):
    python cs117_project/baselines.py
    python cs117_project/baselines.py --cv
Output: results/baseline_test.json (or results/baseline_cv.json with --cv)
"""
import argparse

import numpy as np

from common import CLASSES, RESULTS, TEXT_MODELS, read_splits, run_info, write_json
from evaluation import balanced_acc, load_all_folds, load_split, summarize, text_feats


def run_cv(args):
    out = {"run": run_info(), "classes": CLASSES, "metric": "top-1 accuracy (also balanced accuracy)",
           "protocol": "zero-shot scored on every image in data/folds.csv (each once); per-fold numbers too",
           "results": {"random": {"top1_analytic": round(1 / len(CLASSES), 4)}}}
    for m in args.models:
        X, y, fold = load_all_folds(m)
        T, prompts = text_feats(m)
        pred = (X @ T.T).argmax(1)
        n_folds = int(fold.max()) + 1
        r = {"prompts": prompts, **summarize(y, pred),
             "per_fold_top1": [round(float((pred[fold == f] == y[fold == f]).mean()), 4) for f in range(n_folds)],
             "per_fold_balanced_acc": [round(balanced_acc(y[fold == f], pred[fold == f]), 4) for f in range(n_folds)]}
        out["results"][f"zeroshot_{m}"] = r
        print(f"[all folds] zero-shot {m}: top-1 {r['top1']:.3f} (95% CI {r['top1_ci95']}), balanced "
              f"{r['balanced_acc']:.3f} on {r['n_images']} images; per fold {r['per_fold_top1']}")
    print(f"[all folds] random: {1 / len(CLASSES):.3f} analytic")
    write_json(RESULTS / "baseline_cv.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(TEXT_MODELS), choices=list(TEXT_MODELS))
    ap.add_argument("--random-trials", type=int, default=1000)
    ap.add_argument("--cv", action="store_true", help="score on every image in data/folds.csv")
    args = ap.parse_args()
    if args.cv:
        return run_cv(args)

    splits = [s for s in ("test", "phone") if read_splits(s)]
    out = {"run": run_info(), "classes": CLASSES, "metric": "top-1 accuracy", "splits": {}}
    for split in splits:
        res = {}
        _, y, _ = load_split(args.models[0], split)
        # Random guessing over all 8 classes: analytic chance plus a simulated check.
        rng = np.random.default_rng(0)
        sims = [(rng.integers(0, len(CLASSES), len(y)) == y).mean() for _ in range(args.random_trials)]
        res["random"] = {"top1_analytic": round(1 / len(CLASSES), 4), "top1_simulated_mean": round(float(np.mean(sims)), 4),
                         "top1_simulated_std": round(float(np.std(sims)), 4), "n_images": int(len(y))}
        for m in args.models:
            X, y, _ = load_split(m, split)
            T, prompts = text_feats(m)
            pred = (X @ T.T).argmax(1)
            res[f"zeroshot_{m}"] = {"prompts": prompts, **summarize(y, pred)}
            print(f"[{split}] zero-shot {m}: top-1 {res[f'zeroshot_{m}']['top1']:.3f} "
                  f"(95% CI {res[f'zeroshot_{m}']['top1_ci95']}) on {len(y)} images")
        print(f"[{split}] random: {res['random']['top1_analytic']:.3f} analytic, "
              f"{res['random']['top1_simulated_mean']:.3f} simulated")
        out["splits"][split] = res
    write_json(RESULTS / "baseline_test.json", out)


if __name__ == "__main__":
    main()
