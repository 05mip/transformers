# Project log (Team 11, CS 117)

Running, dated log of what we did, what broke and how we fixed it, failures we have not resolved, and things to
remember for later. Newest entry goes at the top of the log section. The open to-do list and the AI-tool
disclosure live at the bottom. Final write-up claims must come from this log or from results files, not memory.

## Open to-do / things to remember
- [ ] Rerun the smoke test on Colab T4 (the course requires Colab). So far it only ran on a local laptop.
- [x] Proposal section 4 (due Oct 28): pasted real console output.
- [ ] Commit and push `cs117_project/` work to the fork. Do not add a Claude co-author line.
- [ ] CONTRIBUTION.md location/name: check-in handout says `CONTRIBUTIONS.md`; project page says `CONTRIBUTION.md`.
      Read the Canvas "Code Submission Guidelines" before Dec 7 and move/rename to match.
- [ ] Investigate or at least document the failing upstream test `test_inference_interpolate_pos_encoding`.
- [x] Embedding + caching script, baselines, few-shot sweep, plotting (written 2026-11-16).
- [ ] Hand-clean web images (contact sheets -> `data/exclude.csv`), then freeze `data/splits.csv`.
- [ ] Decide whether to keep `hybrid` (thin, noisy source data); drop if < ~70 clean images.
- [ ] Phone photos -> `data/phone/<class>/`, labeled by `data/labeling_criteria.md`; test-only.
- [ ] Check-in (due Tue Nov 17, 11:58pm): fill `checkin_template.pptx`, export PDF, submit on Canvas.
- [ ] Final: README with rerun commands + results table; slides Nov 29; presentation Nov 30/Dec 2; video+code Dec 7.
- [x] Group-aware 4-fold CV (KFOLD_PLAN.md), done 2026-11-18. 32-shot point now included.
- [ ] Team to confirm the CV cap choice (option a, `--max-per-source 15`, drops 44 BMX images from one photographer).
- [ ] New road images (if added) need an `author` in `manual_sources.csv`; then rerun make_splits/embed/CV.
- Caveat for write-up: web-scraped bike images may overlap with the models' pretraining data.

## Log

### 2026-11-18: group-aware 4-fold cross-validation (KFOLD_PLAN.md implemented)
- Replaced the single test/pool split with 4 folds (`make_splits.py --folds 4 --max-per-source 15` ->
  `data/folds.csv`): `StratifiedGroupKFold`, stratified by class, grouped by photographer (manual images grouped
  by the new `author` column). `data/splits.csv` and the check-in results files are unchanged.
- **Cap:** used the plan's recommended option (a), a per-source cap of 15. It drops 44 BMX images, all from
  `thomas bresson` (59 -> 15); no other source exceeds 15. Without it one fold gets 59 BMX test images. Team still
  needs to confirm this choice. Balanced accuracy (mean per-class accuracy) is also reported as a safety check.
- Folds (462 images after cap + dedup, 334 source groups), test images per class:
  ```
  fold   road mountain cruiser bmx tricycle tandem hybrid  total
     0     12       15      17  18       14     19     20    115
     1     13       15      17  17       14     20     19    115
     2     13       15      17  17       15     20     19    116
     3     12       15      18  18       14     19     20    116
  ```
- Checks passed: every image in exactly one fold; no source group in two folds (asserted in `make_splits.py`);
  0 dHash pairs at distance <= 6 across folds.
- Environment: new venv `C:\Users\Justi\cs117_venv` (Python 3.12, torch 2.11.0+cu128, RTX 5060). Re-embedded all
  images and reran the single-split scripts first: zero-shot matched the check-in exactly (69.3 / 95.7); 3 few-shot
  means moved by <= 0.15 points (GPU numeric noise). Check-in results files were restored, not overwritten.
- Zero-shot, pooled over all 462 images: CLIP 67.3% (95% CI 62.8-71.4, balanced 68.2), SigLIP 95.2% (93.3-97.2,
  balanced 95.4), random 14.3%. CLIP sanity check vs single split (69.3%): close, as expected.
- Few-shot linear probe, top-1 mean +- std over 4 folds x 5 draws, k = 1/2/4/8/16/32:
  CLIP 42.0/46.8/61.3/70.1/75.0/79.2; SigLIP 64.9/77.9/86.9/92.7/94.0/95.0; DINOv2 39.2/43.2/54.8/69.1/77.6/83.5.
  Std is 2-9 points. Balanced accuracy is within 1.6 points of top-1 everywhere.
- **32 shots now possible** (smallest training side is road, 37-38 images). At k = 32 the 5 draws overlap heavily,
  so the std there mostly comes from the folds.
- **Ranking stability:** SigLIP ranks first in 4/4 folds at every k (and zero-shot). CLIP beats DINOv2 in 3/4 folds
  at k = 1-2 and 4/4 at k = 4, but DINOv2 beats CLIP in 3/4 folds at k = 8, 16, 32. So SigLIP > others is stable;
  CLIP vs DINOv2 flips around 8 shots.
- SigLIP zero-shot (95.2%) is still as good as its 32-shot probe (95.0%); CLIP's probe passes its zero-shot at
  ~8 shots. CLIP zero-shot is weakest on hybrid (38.5%) and cruiser (53.6%).
- Compared with the check-in split at 16 shots: CLIP 80.1 -> 75.0, SigLIP 95.7 -> 94.0, DINOv2 77.3 -> 77.6.
  The drop is consistent with removing same-session leakage (caveat from 2026-11-17), but the check-in numbers
  came from one 140-image split, so we cannot attribute the gap to leakage alone.
- Outputs: `results/baseline_cv.json`, `results/fewshot_cv.json`, `results/fewshot_cv_curve.png`.

### 2026-11-17: check-in numbers (frozen split)
- Justin added 30 BMX + 30 hybrid (`data/manual_sources.csv`); Anish moved 6 tandem/mountain images out of reject;
  Michael removed weak images. 514 images in `data/raw/`; dHash removed 8 near-duplicates -> 506.
- **Per-source cap removed** (team decision): use every hand-reviewed image. `--max-per-source` now defaults to 0.
  Caveat for write-up: some classes have several photos from one event/photographer, so test and pool may share
  a photo session, which can inflate few-shot accuracy.
- Split: 20 test per class (140); pool road 30, mountain 40, cruiser 49, bmx 94, tricycle 37, tandem 58, hybrid 58.
- Zero-shot: CLIP 69.3% (95% CI 61.4-75.7), SigLIP 95.7% (92.1-98.6), random 14.3%.
- Few-shot linear probe (mean of 5 seeds), k = 1/2/4/8/16:
  CLIP 42.0/55.7/65.6/78.4/80.1; SigLIP 71.3/76.6/87.6/94.6/95.7; DINOv2 40.0/47.4/59.9/72.3/77.3.
- CLIP zero-shot fails mostly on cruiser (45%) and hybrid (45%); hybrid<->road and cruiser->hybrid confusions.

### 2026-11-16: dataset, full pipeline, provisional baseline
**Done**
- Wrote the pipeline: `collect_images.py` (Commons + Openverse), `collect_web.py` (Bing), `contact_sheets.py` +
  `record_review.py` / `review_folders.py` (hand review -> `data/review.csv`), `make_splits.py`, `embed.py`,
  `baselines.py`, `fewshot.py`, `plot_results.py`.
- Downloaded 1,754 candidates; hand review kept 571; per-source cap (8 per photographer/domain per class) and dHash
  dedup leave 391 across 7 classes. Split: 20 test per class (140 test images), rest is the few-shot pool.
- Classes now: road, mountain, cruiser, bmx, tricycle, tandem, hybrid. **Folding dropped** (team decision, scope).
- **Provisional** results (before Michael's manual review pass; rerun after):
  zero-shot CLIP 69.3% (95% CI 61.4-76.4) on 140 test images; zero-shot SigLIP 94.3%; random 14.3%.
  Few-shot (16 shots): CLIP 84.7%, SigLIP 95.1%, DINOv2 83.1%. 1-2 shot probes are BELOW zero-shot for CLIP/SigLIP.

**What broke and how we fixed it**
| # | Problem | Fix |
|---|---|---|
| 7 | Web labels are noisy: only ~1/3 of Commons/Openverse images show one clear bike of the type | Hand review against `data/labeling_criteria.md`; only `keep` images are used |
| 8 | 144/150 BMX images from one photographer at one event (also clusters in tandem, tricycle) -> test/pool leakage | Cap of 8 images per source per class in `make_splits.py` |
| 9 | Wikimedia 429s: non-standard 512px thumbnails are rejected | Use standard 500px width; retry with backoff |
| 10 | icrawler's Google crawler is broken | Use Bing |
| 11 | Bing (no cookies) mostly searches the first query word ("kids tricycle" -> photos of kids) | Queries start with the class noun; junk deleted |
| 12 | SigLIP tokenizer needs `sentencepiece` | `pip install sentencepiece protobuf` |

**Notes for later**
- Hybrid is thin (48 after cap) and BMX thinnest (37): max shots is 16, not 32, until we collect more.
- Bing/manual images are copyrighted: evaluation only, not redistributed; URLs recorded in the manifest.
- Teammates need the images (git-ignored): share `data/raw/` via OneDrive/Drive, or re-download from the manifest URLs.

### 2026-10-27: fork setup, environment, smoke tests
**Done**
- Fork is `github.com/05mip/transformers` (shallow clone, `upstream` remote = huggingface/transformers) at
  `CS117/project/transformers`. HEAD is upstream commit `95bafdc`.
- Created `cs117_project/` with `smoke_test.py`, `CONTRIBUTION.md`, `NOTES.md`. No upstream files changed.
- Local environment: Windows 11, RTX 3060 Laptop GPU, Python 3.12.1. Fresh venv at `C:\Users\micha\cs117_venv` with
  torch 2.6.0+cu124 and the fork installed with `pip install -e .`.

**Smoke test 1: our script** (`python cs117_project/smoke_test.py`), exit 0:
```
python 3.12.1 | torch 2.6.0+cu124 | transformers 5.21.0.dev0 | device cuda (NVIDIA GeForce RTX 3060 Laptop GPU)
transformers imported from: ...\project\transformers\src\transformers\__init__.py
OK openai/clip-vit-base-patch16: embedding (1, 512), L2 norm 1.000
OK google/siglip-base-patch16-224: embedding (1, 768), L2 norm 1.000
OK facebook/dinov2-base: embedding (1, 768), L2 norm 1.000
```
Useful: CLIP embeds to 512 dims, SigLIP and DINOv2 to 768. The "imported from" line shows the fork's own code is used.

**Smoke test 2: the repo's own tests**
`RUN_SLOW=1 pytest tests/models/clip/test_modeling_clip.py -k IntegrationTest -v` gave 1 passed, 1 failed.
- PASSED: `CLIPModelIntegrationTest::test_inference` (loads `openai/clip-vit-base-patch32`, not our patch16).
- FAILED: `test_inference_interpolate_pos_encoding`. Output mismatch, max abs difference 0.0030 vs 0.0004 allowed.
  Identical on CPU (`CUDA_VISIBLE_DEVICES=-1`) and GPU, so it is not GPU precision. Cause not found. Candidates:
  stale upstream expected values, or a different Pillow or torchvision version. We changed no upstream files.
  Unresolved; do not claim a cause in the write-up.

**What broke and how we fixed it**
| # | Problem | Fix |
|---|---|---|
| 1 | Global torch 2.5.1; transformers 5.x prints "PyTorch >= 2.6 is required" and disables torch, so `AutoModel` raises ImportError | New venv with torch 2.6.0 from the cu124 wheel index |
| 2 | A venv that reused system site-packages plus a newer torch: `DLL load failed while importing _C` | Use a clean venv with no system packages (and a short path) |
| 3 | `get_image_features` returns a ModelOutput in this version, so `F.normalize` crashed with `no attribute 'norm'` | `smoke_test.py` falls back to `.pooler_output` if the result is not a tensor |
| 4 | pytest crashed: `unknown hook 'pytest_configure_node'` | `pip install pytest-xdist` (the repo's conftest needs it) |
| 5 | Collecting the CLIP tests failed: `No module named 'datasets'` | `pip install datasets pytest-env pytest-rerunfailures timeout-decorator`, plus `pytest pytest-timeout parameterized psutil accelerate` |
| 6 | Harmless warnings: unauthenticated HF Hub requests; no symlink support on Windows; torchvision missing for two image processors | Ignored. Silence with `HF_TOKEN`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, installing torchvision |

**Notes for later**
- `pip install -e ".[testing]"` would likely pull all test deps at once; we installed them piecemeal.
- Colab will differ: its own torch version may already satisfy the 2.6 requirement. Check before installing.
- The repo's `CLAUDE.md` points to `.ai/AGENTS.md`. Not read yet; may hold upstream contributor rules.

## AI-tool use disclosure
- Claude Code (Sonnet 5.5) generated `smoke_test.py`, `CONTRIBUTION.md` and this log, and ran the environment setup
  and smoke tests under the user's direction.
- Mistakes it made (log every one):
  1. First `smoke_test.py` assumed `get_image_features` returns a tensor (problem 3).
  2. Did not check the torch version requirement before the first run (problem 1).
  3. Tried a system-site-packages venv first, which broke torch (problem 2).
  4. First collector used a non-standard 512px thumbnail width and discarded every image that got a 429 (problem 9).
  5. Assumed icrawler URL-encodes queries; it doesn't, and Bing still weights the first word, so two download
     batches were mostly junk (problem 11). Caught by looking at the contact sheets (e.g. "carbon road bike" ->
     atom diagrams).
  6. Saved the manifest only after each class finished, so stopping the first run mid-class lost the road rows.
- Claude Code (Opus 5.5) did the first-pass hand review from contact sheets; Michael re-checks it manually.
