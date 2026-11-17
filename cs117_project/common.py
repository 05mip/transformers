"""Shared constants and helpers for the CS117 bike few-shot project."""
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PHONE_DIR = DATA / "phone"          # our own phone photos: data/phone/<class>/*.jpg (test-only)
SPLITS = DATA / "splits.csv"        # committed: file,label,split (split = test | pool | phone)
REVIEW = DATA / "review.csv"        # committed: file,label,decision (keep|reject) from hand review
FEATURES = ROOT / "features"        # git-ignored embedding cache
RESULTS = ROOT / "results"          # committed outputs

# "folding" dropped 2026-11-16 (team decision to cut scope; often only recognizable by a fold hinge).
CLASSES = ["road", "mountain", "cruiser", "bmx", "tricycle", "tandem", "hybrid"]

# Zero-shot prompt, fixed BEFORE any test scoring (no prompt tuning on test data).
PROMPT_TEMPLATE = "a photo of a {}."
PROMPT_NAMES = {
    "road": "road bike", "mountain": "mountain bike", "cruiser": "cruiser bike", "bmx": "BMX bike",
    "tricycle": "tricycle", "folding": "folding bike", "hybrid": "hybrid bike", "tandem": "tandem bike",
}

MODELS = {  # short name -> Hub id. All ViT-B scale.
    "clip": "openai/clip-vit-base-patch16",
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
}
TEXT_MODELS = ("clip", "siglip")  # DINOv2 has no text encoder -> no zero-shot


def image_path(file):
    """splits.csv paths are relative to data/raw (web) or data/ (phone/...)."""
    return DATA / file if file.startswith("phone/") else RAW_DIR / file


def read_splits(split=None):
    rows = list(csv.DictReader(SPLITS.open(newline="", encoding="utf-8")))
    return [r for r in rows if split is None or r["split"] == split]


def run_info():
    """Provenance stamped into every results file."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    import torch
    import transformers
    return {
        "command": "python " + " ".join([Path(sys.argv[0]).as_posix().split("transformers/", 1)[-1]] + sys.argv[1:]),
        "git_commit": commit,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT.parent).as_posix()}")
