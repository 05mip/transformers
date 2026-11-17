"""Collect CC-licensed bike-type images from Wikimedia Commons (primary) and Openverse (top-up).

Images go to cs117_project/data/raw/<class>/ (git-ignored). Every downloaded file gets a row in
cs117_project/data/manifest_raw.csv with its source, page URL, license and author, so the dataset
can be rebuilt from the manifest alone.

Usage:
    python cs117_project/collect_images.py --per-class 150
"""
import argparse
import csv
import hashlib
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest_raw.csv"
UA = {"User-Agent": "cs117-course-project-bike-dataset/0.1 (educational, non-commercial)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
THUMB_WIDTH = 500  # a Wikimedia standard thumbnail width (non-standard sizes get 429s)

# Commons categories per class: (category, include_subcategories_one_level_down)
COMMONS = {
    "road": [("Road bicycles", False), ("Road racing bicycles", True)],
    "mountain": [("Mountain bikes", True)],
    "cruiser": [("Cruiser bicycles", False), ("Beach cruisers", False)],
    "bmx": [("BMX bicycles", False), ("Freestyle BMX", False), ("Haro bicycles", False)],
    "tricycle": [("Tricycles", False), ("Children's tricycles", True)],
    "folding": [("Folding bicycles", True)],
    "hybrid": [("Hybrid bikes", True)],
    "tandem": [("Tandem bicycles", False), ("Tandem racing", False)],
}
# Subcategories we never descend into (not the class, or not photos of the bike).
SKIP_SUBCAT = re.compile(r"part|video|art\b|in art|drawing|logo|diagram|patent|motor|electric|e-|"
                         r"time trial|triathlon|triples|quads|quints", re.I)
# Openverse search terms per class, used only to top up classes short on Commons images.
OPENVERSE = {
    "road": ["road bike"], "mountain": ["mountain bike"], "cruiser": ["beach cruiser bicycle", "cruiser bike"],
    "bmx": ["bmx bike"], "tricycle": ["tricycle"], "folding": ["folding bike"],
    "hybrid": ["hybrid bike", "hybrid bicycle"], "tandem": ["tandem bike", "tandem bicycle"],
}
FIELDS = ["file", "label", "source", "source_id", "page_url", "image_url", "license", "author", "sha1"]


def get_json(url, params, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, headers=UA, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        time.sleep(5 * (i + 1))  # rate-limited or transient error: back off
    raise RuntimeError(f"failed after {tries} tries: {url} {params} -> {r.status_code}")


def commons_files(category, with_subcats):
    """Yield (title, thumb_url, page_url, license, author) for files in a category."""
    cats = [category]
    if with_subcats:
        data = get_json(COMMONS_API, {"action": "query", "list": "categorymembers", "format": "json",
                                      "cmtitle": f"Category:{category}", "cmtype": "subcat", "cmlimit": 500})
        cats += [m["title"].removeprefix("Category:") for m in data["query"]["categorymembers"]
                 if not SKIP_SUBCAT.search(m["title"])]
    for cat in cats:
        params = {"action": "query", "format": "json", "generator": "categorymembers",
                  "gcmtitle": f"Category:{cat}", "gcmtype": "file", "gcmlimit": 200,
                  "prop": "imageinfo", "iiprop": "url|mime|extmetadata", "iiurlwidth": THUMB_WIDTH}
        while True:
            data = get_json(COMMONS_API, params)
            for page in data.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                meta = info.get("extmetadata", {})
                author = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
                yield (page["title"], info.get("thumburl") or info["url"], info.get("descriptionurl", ""),
                       meta.get("LicenseShortName", {}).get("value", ""), author[:120])
            if "continue" not in data:
                break
            params.update(data["continue"])
            time.sleep(1)
        time.sleep(1)


def openverse_files(query, max_pages=12):
    for page in range(1, max_pages + 1):
        data = get_json(OPENVERSE_API, {"q": query, "page_size": 20, "page": page,
                                        "license_type": "all-cc", "mature": "false"})
        for r in data.get("results", []):
            yield (r["id"], r["url"], r.get("foreign_landing_url", ""),
                   f"{r.get('license', '')} {r.get('license_version', '')}".strip(), (r.get("creator") or "")[:120])
        if page >= data.get("page_count", 0):
            break
        time.sleep(3)


def download(url, tries=6):
    """GET with backoff on 429/5xx, honoring Retry-After."""
    for i in range(tries):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 0) or 0) or 10 * (i + 1)
            print(f"  {r.status_code}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up after {tries} tries: {url}")


def save_image(content, path):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if min(img.size) < 128:
        raise ValueError("too small")
    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
    img.save(path, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150, help="max images to download per class")
    ap.add_argument("--classes", nargs="*", default=list(COMMONS))
    args = ap.parse_args()

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8"))) if MANIFEST.exists() else []
    seen_ids = {(r["source"], r["source_id"]) for r in rows}
    seen_sha = {r["sha1"] for r in rows}

    for label in args.classes:
        out = RAW_DIR / label
        out.mkdir(parents=True, exist_ok=True)
        have = sum(r["label"] == label for r in rows)
        sources = [("commons", commons_files(c, s)) for c, s in COMMONS[label]]
        sources += [("openverse", openverse_files(q)) for q in OPENVERSE[label]]
        for source, gen in sources:
            for source_id, url, page_url, lic, author in gen:
                if have >= args.per_class:
                    break
                if (source, source_id) in seen_ids:
                    continue
                seen_ids.add((source, source_id))
                try:
                    content = download(url)
                    sha = hashlib.sha1(content).hexdigest()
                    if sha in seen_sha:
                        continue
                    fname = f"{label}_{source}_{sha[:12]}.jpg"
                    save_image(content, out / fname)
                except Exception as e:  # broken link, unreadable image, too small
                    print(f"  skip {source} {source_id}: {e}")
                    continue
                seen_sha.add(sha)
                rows.append(dict(file=f"{label}/{fname}", label=label, source=source, source_id=source_id,
                                 page_url=page_url, image_url=url, license=lic, author=author, sha1=sha))
                have += 1
                time.sleep(1.0)
            if have >= args.per_class:
                break
        print(f"{label}: {have} images")
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:  # checkpoint after each class
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
