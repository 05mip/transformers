# K-fold cross-validation plan

What it would take to replace the single frozen test/pool split with group-aware k-fold CV, as
promised on the check-in slides (items 3b and 4b: "Michael, Nov 24"). Written 2026-11-18 from reading
the current code and data.

**Status (2026-11-18): implemented and run** with option (a), `--max-per-source 15`. Results and checks are
logged in `NOTES.md` (entry 2026-11-18); outputs are `results/*_cv.json` and `results/fewshot_cv_curve.png`.

## Why

- One 20-per-class test split gives noisy numbers (CLIP zero-shot CI is 61.4-75.7%, per-class scores
  move in 5-point steps).
- The research question asks whether the model ranking is **stable across splits**. One split can't
  answer that.
- With CV every image is tested exactly once, so per-class accuracy is computed over all 50-114 images
  per class instead of 20.

## The protocol

1. Assign every image in `data/raw/` to one of **K = 4 folds**: stratified by class and **grouped by
   source** (photographer or site), so a photo session never spans train and test.
2. For each fold f: test = fold f, few-shot pool = the other 3 folds.
3. For each model, k in {1, 2, 4, 8, 16, 32}, seed in {0..4}: draw k pool images per class, fit the
   same logistic regression as now, score on fold f.
4. Report **mean ± std over 4 folds × 5 seeds** (20 runs per point). Also report per-fold model ranks
   to answer the "is the ranking stable" question.
5. Zero-shot (CLIP, SigLIP) needs no training, so it is scored on all 506 images at once (each image
   tested once) with the existing bootstrap CI. Per-fold zero-shot numbers are also stored, for the
   ranking comparison.

## Things found in the data that the plan has to handle

These came from simulating `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)` on the
current 506 images, using the photographer groups from item 2 (rerun 2026-11-18).

### 1. One BMX photographer is half the class

`thomas bresson` has **59 of 114** BMX images. Grouping keeps them together, so one fold gets 59 BMX
test images and the others get about 18:

```
no cap:  fold 2 test per class [12, 15, 17, 59, 14, 19, 19]   <- BMX swamps this fold
cap 15:  every fold            [12-13, 15, 17-18, 17-18, 14-15, 19-20, 19-20]
```

Options (team decision needed, since the cap was removed on 2026-11-17):

- **(a) Recommended:** bring back a per-source cap only for huge groups (`--max-per-source 15`).
  It drops 44 images (all from oversized groups) and gives evenly balanced folds.
- (b) Keep everything and report **balanced accuracy** (mean of per-class accuracy) instead of
  plain top-1, since folds are no longer class-balanced. Should be done anyway as a safety measure.
- (c) Keep that photographer's images in the pool only (never test). Simple, but less clean to explain.

### 2. Hand-added images were grouped by hosting site, not photographer (DONE, commit 647e295)

`make_splits.py` used the URL's domain as the "source" for images in `manual_sources.csv`, which made
fake groups: 21 BMX images became one group `thumb.wikimedia.org`, 27 hybrid images became
`live.staticflickr.com`.

Fixed by Justin on 2026-11-18: `data/manual_sources.csv` now has an `author` column (Commons artist,
`flickr:<account>`, or `Science Museum Group`; one bot-uploaded image left blank), and `make_splits.py`
reads it. A blank author makes the image its own group. The largest manual-image group is now 3 photos,
so manual images no longer unbalance folds. The fold counts above were rerun with these groups.

### 3. The 32-shot point is possible without new images

With 4 folds the smallest training side is road: 50 × 3/4 ≈ **37-38 images**, which is ≥ 32. So the
32-shot point that was blocked (check-in item 4a.1) works under CV with the data we already have.
Caveat: at k = 32 out of 37, the 5 seed draws overlap heavily, so the std at 32 shots comes mostly from
folds rather than draws. Say so in the write-up. More road images still help.

### 4. Near-duplicates are already handled

`make_splits.py` already drops all but one image per dHash near-duplicate group, so the only grouping
the folds need is by source. Keep running dedup **before** the fold assignment.

## Code changes, file by file

Embeddings are cached per file and don't depend on the split, so **no GPU work is needed** beyond
re-embedding any newly added images. The whole CV sweep is about 360 logistic-regression fits on cached
features (a few minutes on CPU).

| File | Change |
|---|---|
| `make_splits.py` | Add `--folds 4`. After dedup (and optional cap), build `groups = class + source`, run `sklearn.model_selection.StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)`, and write **`data/folds.csv`** (`file,label,source,fold`). (Reading the `author` column from `manual_sources.csv` is already done.) Print a per-fold × per-class count table. Leave `splits.csv` untouched so the check-in numbers stay reproducible. |
| `common.py` | Add `FOLDS = DATA / "folds.csv"` and `read_folds()` returning rows with an int `fold`. |
| `embed.py` | Embed the union of files in `splits.csv` and `folds.csv` (today it only reads `splits.csv`), so new images get features. |
| `evaluation.py` | Add `load_fold(model, fold)` returning `(X_train, y_train), (X_test, y_test)` from `folds.csv`. Add `balanced_acc` (mean per-class accuracy) to `summarize`. |
| `fewshot.py` | Add `--cv`. Outer loop over folds; draw shots from the train folds; skip k if any class's train side has fewer than k images *in that fold*. Store per-fold and per-seed accuracies plus mean/std over all 20 runs, and a **pooled confusion matrix** summed over folds (each image counted once). Write `results/fewshot_cv.json`. |
| `baselines.py` | Add `--cv`: zero-shot scored per fold and pooled over all images; random baseline unchanged (1/7). Write `results/baseline_cv.json`. |
| `plot_results.py` | Read the CV json, y-label "top-1 accuracy (mean ± std, 4 folds × 5 draws)", add the 32 tick. Optional second figure: model rank per fold. |
| `NOTES.md` | Dated log entry: fold counts, the cap decision, results. Update the open to-do list. |
| `CONTRIBUTION.md` | Add `data/folds.csv` and the new results files. |

Sketch of the fold assignment (inside `make_splits.py`, after dedup):

```python
from sklearn.model_selection import StratifiedGroupKFold

files  = [f for c in CLASSES for f in per_class[c]]
labels = [f.split("/")[0] for f in files]
groups = [f"{lab}|{source.get(f.split('/')[-1]) or f}" for f, lab in zip(files, labels)]  # unknown -> own group
sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
fold_of = {}
for k, (_, test_idx) in enumerate(sgkf.split(files, labels, groups)):
    for i in test_idx:
        fold_of[files[i]] = k
```

## Checks before trusting the numbers

- Every image appears in exactly one fold; `folds.csv` has the same 506 (or capped) files as `data/raw/`.
- No source group appears in two folds (assert in `make_splits.py`).
- No dHash pair at distance ≤ 6 between any test fold and its train folds.
- Per-fold × per-class count table printed and pasted into `NOTES.md`.
- Sanity: pooled zero-shot CLIP on all images should land near the single-split 69.3%. A big jump
  means something is off.
- The few-shot classifier never sees test-fold images (C is still chosen by CV *inside* the k-shot
  training set only).

## Order of work

1. Team decides the cap question (option a, b or c above). Takes 5 minutes in chat.
2. ~~Justin adds the `author` column to `manual_sources.csv`~~ (done, commit 647e295). Any new road
   images added later need an `author` filled in too.
3. Change `make_splits.py` and `common.py`, generate `folds.csv`, check fold counts.
4. Run `embed.py` if any images were added.
5. Change `evaluation.py`, `baselines.py`, `fewshot.py`, `plot_results.py`; run them.
6. Log in `NOTES.md`; update slides.

Commands once implemented:

```
python cs117_project/make_splits.py --folds 4 --max-per-source 15
python cs117_project/embed.py            # only if images were added
python cs117_project/baselines.py --cv
python cs117_project/fewshot.py --cv --shots 1 2 4 8 16 32
python cs117_project/plot_results.py --cv
```

## What to say in the write-up

- Each image is tested exactly once; numbers are mean ± std over 4 group-aware folds × 5 shot draws.
- Folds group images by photographer/site, so a photo session never spans train and test (this
  addresses the leakage caveat in `NOTES.md`).
- If the cap is used, state how many images it dropped and from which sources.
- Ranking stability: report in how many of the 4 folds each model ranks first at each k.
