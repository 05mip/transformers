"""Embed every image in data/splits.csv and data/folds.csv with a frozen model; cache L2-normalized features.

Output: features/<model>.npz with arrays `files` (str) and `feats` (float32, N x D), plus for text
models `text_feats` (C x D, one row per class in common.CLASSES, from the fixed zero-shot prompt).

Usage:
    python cs117_project/embed.py --models clip siglip dinov2
"""
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common import CLASSES, FEATURES, FOLDS, MODELS, PROMPT_NAMES, PROMPT_TEMPLATE, TEXT_MODELS, image_path, read_folds, read_splits


def as_tensor(out):
    """get_*_features returns a tensor in older transformers, a ModelOutput in 5.x."""
    return out if torch.is_tensor(out) else out.pooler_output


@torch.no_grad()
def embed_images(name, model, processor, files, device, batch_size):
    feats = []
    for i in range(0, len(files), batch_size):
        imgs = [Image.open(image_path(f)).convert("RGB") for f in files[i:i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        if name in TEXT_MODELS:
            f = as_tensor(model.get_image_features(**inputs))
        else:  # DINOv2: CLS token of the last hidden state (decided up front, not tuned)
            f = model(**inputs).last_hidden_state[:, 0]
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        print(f"  {name}: {min(i + batch_size, len(files))}/{len(files)}", end="\r")
    print()
    return torch.cat(feats).numpy()


@torch.no_grad()
def embed_text(name, model, processor, device):
    prompts = [PROMPT_TEMPLATE.format(PROMPT_NAMES[c]) for c in CLASSES]
    # SigLIP was trained with max_length padding; other padding gives wrong text embeddings.
    pad = "max_length" if name == "siglip" else True
    inputs = processor(text=prompts, padding=pad, return_tensors="pt").to(device)
    f = as_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy(), prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = [r["file"] for r in read_splits()]
    if FOLDS.exists():  # union: CV folds may keep a different near-duplicate representative than splits.csv
        seen = set(files)
        files += [r["file"] for r in read_folds() if r["file"] not in seen]
    FEATURES.mkdir(exist_ok=True)
    for name in args.models:
        processor = AutoProcessor.from_pretrained(MODELS[name])
        model = AutoModel.from_pretrained(MODELS[name]).to(device).eval()
        out = {"files": np.array(files), "feats": embed_images(name, model, processor, files, device, args.batch_size)}
        if name in TEXT_MODELS:
            out["text_feats"], prompts = embed_text(name, model, processor, device)
            out["prompts"] = np.array(prompts)
        np.savez(FEATURES / f"{name}.npz", **out)
        print(f"{name}: {out['feats'].shape} -> features/{name}.npz")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
