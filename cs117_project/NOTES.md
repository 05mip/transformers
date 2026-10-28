# Project log (Team 11, CS 117)

Running, dated log of what we did, what broke and how we fixed it, failures we have not resolved, and things to
remember for later. Newest entry goes at the top of the log section. The open to-do list and the AI-tool
disclosure live at the bottom. Final write-up claims must come from this log or from results files, not memory.

## Open to-do / things to remember
- [ ] Rerun the smoke test on Colab T4 (the course requires Colab). So far it only ran on a local laptop.
- [ ] Proposal section 4 (due Oct 28): paste real console output; see "2026-10-27" for what we have.
- [ ] Commit and push `cs117_project/` to the fork (nothing committed yet). Do not add a Claude co-author line.
- [ ] Decide where CONTRIBUTION.md should live: currently `cs117_project/CONTRIBUTION.md`; the course may expect
      it at the repo root. Check the syllabus.
- [ ] Investigate or at least document the failing upstream test `test_inference_interpolate_pos_encoding`.
- [ ] Next build step: shared embedding + caching script for the three models (Oct 30), then few-shot sweep and plots.
- Caveat for write-up: web-scraped bike images may overlap with the models' pretraining data.

## Log

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
