"""Plot the main result: top-1 accuracy vs. shots per class, one line per model, one panel per test set.

Usage (after baselines.py and fewshot.py):
    python cs117_project/plot_results.py
    python cs117_project/plot_results.py --cv     # after fewshot.py --cv
Output: results/fewshot_curve.png (or results/fewshot_cv_curve.png with --cv)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CLASSES, RESULTS

COLORS = {"clip": "#2a6fdb", "siglip": "#e07b00", "dinov2": "#2a9d55"}
LABELS = {"clip": "CLIP ViT-B/16", "siglip": "SigLIP B/16", "dinov2": "DINOv2-B"}
TITLES = {"test": "Web images (held-out test)", "phone": "Our phone photos (test only)",
          "cv": "Web images, 4-fold group-aware CV (each image tested once)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="plot results/fewshot_cv.json")
    args = ap.parse_args()
    src, dst = ("fewshot_cv.json", "fewshot_cv_curve.png") if args.cv else ("fewshot_test.json", "fewshot_curve.png")
    fs = json.loads((RESULTS / src).read_text())
    if args.cv:
        p = fs["protocol"]
        ylabel = f"top-1 accuracy (mean ± std, {p['folds']} folds × {len(p['seeds'])} draws)"
    else:
        ylabel = "top-1 accuracy (mean ± std, 5 draws)"
    splits = list(next(iter(fs["results"].values())).keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for m, res in fs["results"].items():
            pts = sorted((int(k), v["mean"], v["std"]) for k, v in res[split].items())
            # plot 0-shot at x=0.5 so it fits on a log2 axis
            xs = [k if k > 0 else 0.5 for k, _, _ in pts]
            ax.errorbar(xs, [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", capsize=3,
                        color=COLORS.get(m), label=LABELS.get(m, m))
        ax.axhline(1 / len(CLASSES), color="gray", ls="--", label=f"random (1/{len(CLASSES)})")
        ax.set_xscale("log", base=2)
        ticks = [0.5] + [int(k) for k in fs["protocol"]["shots"]]
        ax.set_xticks(ticks, ["0"] + [str(t) for t in ticks[1:]])
        ax.set_ylim(0, 1)
        ax.set_xlabel("labeled examples per class (shots)")
        ax.set_ylabel(ylabel)
        ax.set_title(TITLES.get(split, split))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / dst, dpi=150)
    print(f"wrote results/{dst}")


if __name__ == "__main__":
    main()
