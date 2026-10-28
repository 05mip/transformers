"""Smoke test: load CLIP, SigLIP, DINOv2 from the Hub and embed one image each."""
import sys
import requests
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODELS = [
    "openai/clip-vit-base-patch16",
    "google/siglip-base-patch16-224",
    "facebook/dinov2-base",
]
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"transformers {transformers.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print("transformers imported from:", transformers.__file__)
    image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=30).raw).convert("RGB")

    for name in MODELS:
        processor = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):  # CLIP, SigLIP
                feats = model.get_image_features(**inputs)
                if not torch.is_tensor(feats):  # newer versions return a ModelOutput
                    feats = feats.pooler_output
            else:  # DINOv2: CLS token of last hidden state
                feats = model(**inputs).last_hidden_state[:, 0]
        feats = torch.nn.functional.normalize(feats, dim=-1)
        print(f"OK {name}: embedding {tuple(feats.shape)}, L2 norm {feats.norm().item():.3f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
