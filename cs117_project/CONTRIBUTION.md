# CONTRIBUTION

Team 11, CS 117. Everything we wrote lives in `cs117_project/`. No upstream
`transformers` source files are modified (check with `git diff upstream/main --stat`).
Original repository: https://github.com/huggingface/transformers

| File | Why |
|---|---|
| `cs117_project/smoke_test.py` | Loads CLIP, SigLIP, DINOv2 from the Hub and embeds one image, to verify the fork installs and runs. |
| `cs117_project/collect_images.py` | Downloads CC-licensed bike images per type from Wikimedia Commons (+ Openverse top-up) and records source, license and author in `data/manifest_raw.csv`. |
| `cs117_project/contact_sheets.py` | Numbered thumbnail grids of the raw images, used for hand-cleaning. |
| `cs117_project/make_splits.py` | Drops hand-rejected images, removes near-duplicates (dHash), and fixes the test / few-shot-pool split in `data/splits.csv`. |
| `cs117_project/common.py` | Shared constants: classes, fixed zero-shot prompt, model IDs, paths, run provenance. |
| `cs117_project/embed.py` | Embeds every split image with each frozen model and caches the features. |
| `cs117_project/evaluation.py` | Loads cached features per split; top-1 accuracy with bootstrap CI, per-class accuracy, confusion matrix. |
| `cs117_project/baselines.py` | Random-guess and zero-shot CLIP/SigLIP baselines -> `results/baseline_test.json`. |
| `cs117_project/fewshot.py` | k-shot logistic-regression sweep over models, shots and seeds -> `results/fewshot_test.json`. |
| `cs117_project/plot_results.py` | Accuracy-vs-shots curve -> `results/fewshot_curve.png`. |
| `cs117_project/data/manifest_raw.csv` | Every downloaded image with its source URL and license, so the dataset can be rebuilt. |
| `cs117_project/data/exclude.csv` | Hand-cleaning decisions (file, reason). |
| `cs117_project/data/splits.csv` | The frozen split (test / pool / phone). |
| `cs117_project/data/labeling_criteria.md` | One-line rule per bike type and the reject reasons. |
| `cs117_project/results/*` | Saved outputs behind every reported number. |
| `cs117_project/CONTRIBUTION.md` | This file. |
| `cs117_project/NOTES.md` | Dated project log: what we did, what broke and the fixes, open failures, to-dos, and the AI-tool use disclosure. |
