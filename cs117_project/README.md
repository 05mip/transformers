# CS117 Final Project – Team 11

## Few-shot bike type classification

This project measures how well frozen image models distinguish 7 visually similar bike types: road, mountain, cruiser, BMX, tricycle, tandem and hybrid.

For each model, we train a logistic regression classifier on its image embeddings using a small number of labeled examples per class (0, 1, 2, 4, 8, 16 or 32) and measure how accuracy changes as more examples are added. We also test whether the ranking of the models is consistent across different train/test splits.

**Models:** CLIP ViT-B/16, SigLIP B/16 and DINOv2-B, all frozen (no fine-tuning).

**Metric:** top-1 accuracy under 4-fold cross-validation. Each image is tested exactly once, and results are reported as mean ± std over 4 folds × 5 random draws of the training examples. Balanced accuracy (the mean of the per-class accuracies) is reported as well.

All project code is in `cs117_project/`. No files from the original transformers library were modified.

- Fork: https://github.com/05mip/transformers
- Original repository: https://github.com/huggingface/transformers

---

## Setup

All commands are run from the repository root (the `transformers/` folder). Python 3.12 is recommended.

```bash
pip install -e .
pip install torch numpy pillow scikit-learn matplotlib requests icrawler
```

The first command installs this fork of transformers (results were produced with version 5.21.0.dev0).

To verify the installation:

```bash
python cs117_project/smoke_test.py
```

This loads all three models and embeds a single test image.

A GPU is optional. Without one, the embedding step runs on the CPU and takes longer.

---

## Reproducing the results

The dataset (`data/raw/`) and fold assignments (`data/folds.csv`) are included in the repository. To reproduce the reported numbers exactly, use the committed `data/folds.csv` and begin at step 2. The "Rebuilding the dataset" section is only needed to collect the images again from scratch.

**1. Create the folds (optional)**

```bash
python cs117_project/make_splits.py --folds 4 --max-per-source 15
```

Writes `data/folds.csv`. Skip this step when reproducing the reported results: rerunning it keeps the same 462 images and photographer groups but may assign them to different folds, which changes the numbers slightly.

**2. Compute embeddings**

```bash
python cs117_project/embed.py
```

Saves features to `features/` (not committed).

**3. Run the baselines (random guess and zero-shot)**

```bash
python cs117_project/baselines.py --cv
```

Writes `results/baseline_cv.json`.

**4. Run the few-shot experiments**

```bash
python cs117_project/fewshot.py --cv --shots 1 2 4 8 16 32
```

Writes `results/fewshot_cv.json`.

**5. Make the plot**

```bash
python cs117_project/plot_results.py --cv
```

Writes `results/fewshot_cv_curve.png`.

**Note:** steps 3-5 overwrite the committed results files, and step 1 overwrites `data/folds.csv`. To restore the committed versions after a test run:

```bash
git checkout -- cs117_project/data/folds.csv cs117_project/results
```

### Rebuilding the dataset (optional)

```bash
python cs117_project/collect_images.py   # Wikimedia Commons + Openverse
python cs117_project/collect_web.py      # Bing image search
```

New images are saved to `data/review_view/<class>/unreviewed/`. Each image is reviewed by hand against `data/labeling_criteria.md`, and accepted images are moved into `data/raw/<class>/`.

### Older single-split results

Before moving to cross-validation, the project used a single fixed test split (20 images per class). Those results are saved in `results/*_test.json` and `results/fewshot_curve.png`, and can be reproduced by running the same scripts without `--cv`:

```bash
python cs117_project/make_splits.py --n-test 20
python cs117_project/baselines.py
python cs117_project/fewshot.py --shots 1 2 4 8 16
python cs117_project/plot_results.py
```

---

## Results

### Baseline

The baseline is zero-shot CLIP, which uses no labeled examples, only the text prompt "a photo of a {bike type}." It was evaluated on all 462 images.

| Method           | Top-1 (%) | 95% CI    | Balanced acc. (%) |
| ---------------- | --------- | --------- | ----------------- |
| Random guess     | 14.3      | –         | 14.3              |
| Zero-shot CLIP   | 67.3      | 62.8–71.4 | 68.2              |
| Zero-shot SigLIP | 95.2      | 93.3–97.2 | 95.4              |

Source: `results/baseline_cv.json`

### Accuracy vs. number of labeled examples

Top-1 accuracy (%), mean ± std. The 0-shot column uses the zero-shot prompt; DINOv2 has no text encoder and therefore no 0-shot result.

| Shots per class | 0          | 1          | 2          | 4          | 8          | 16         | 32         |
| --------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| CLIP            | 67.3 ± 4.0 | 42.0 ± 7.4 | 46.8 ± 6.6 | 61.3 ± 6.0 | 70.1 ± 4.9 | 75.0 ± 5.2 | 79.2 ± 3.8 |
| SigLIP          | 95.2 ± 1.0 | 64.9 ± 4.9 | 77.9 ± 8.4 | 86.9 ± 4.4 | 92.7 ± 2.5 | 94.0 ± 2.5 | 95.0 ± 1.6 |
| DINOv2          | –          | 39.2 ± 8.6 | 43.2 ± 8.5 | 54.8 ± 5.1 | 69.1 ± 4.6 | 77.6 ± 5.8 | 83.5 ± 4.4 |

Source: `results/fewshot_cv.json`; plot: `results/fewshot_cv_curve.png`. The JSON files also include per-class accuracy, confusion matrices and per-fold model rankings.

---

## Data

- 539 hand-reviewed images across 7 classes, stored in `data/raw/<class>/`
- The reported results use 462 images: the 514 in `data/raw/` at the time, after removing near-duplicates and limiting each photographer to 15 images per class. 25 road images were added afterwards (Nov 18) and will be included in the next run
- Folds are grouped by photographer, so images from the same photographer never appear in both training and test
- Sources are recorded for every image: URL, license and author in `data/manifest_raw.csv` (downloaded images), and URL and photographer in `data/manual_sources.csv` (manually added images)

---

## External resources

**Models (from Hugging Face):**

- CLIP: [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
- SigLIP: [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- DINOv2: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base)

**Image sources:** Wikimedia Commons, Openverse, Bing image search (via `icrawler`) and manually selected web images

**Libraries:** PyTorch, transformers, scikit-learn (logistic regression), matplotlib

---

## Files

See `CONTRIBUTION.md` for a description of each file.
