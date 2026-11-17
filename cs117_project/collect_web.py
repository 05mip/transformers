"""Top up classes with images from Bing image search (Google's crawler is broken in icrawler).

Images go to data/raw/<class>/<class>_bing_<sha>.jpg; each gets a manifest_raw.csv row with the
source image URL, the query, and the source domain (in the `author` column, used by the per-source
cap in make_splits.py). Web-search images are copyrighted by their owners: we use them for evaluation
only and never redistribute them (data/raw is git-ignored; only URLs are committed).

Usage:
    python cs117_project/collect_web.py --classes bmx tandem tricycle --per-query 80
"""
import argparse
import csv
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote_plus, urlparse

from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler
from PIL import Image

from collect_images import FIELDS, MANIFEST, RAW_DIR, THUMB_WIDTH

QUERIES = {
    # Bing (no cookies) largely searches the FIRST word, so every query starts with the class noun.
    "bmx_v2": ["bmx bicycle", "bmx bike black", "bmx bike chrome", "bmx bike 20 inch", "bmx bikes for sale",
               "bmx freestyle bike parked", "bmx bicycle blue", "bmx bike red", "bmx cruiser 24 inch", "bmx bike kids"],
    "bmx": ["bmx bike", "bmx bicycle side view", "20 inch bmx bike", "bmx bike parked outside", "freestyle bmx bike",
            "bmx race bike", "kids bmx bike", "old school bmx bike", "bmx bike skatepark", "bmx bike street"],
    "tandem": ["tandem bicycle", "tandem bike parked", "couple riding tandem bike", "modern tandem bicycle",
               "tandem bike side view", "road tandem bike", "vintage tandem bicycle", "family tandem bike"],
    "tricycle": ["kids tricycle", "toddler tricycle outdoors", "adult tricycle", "adult trike three wheel bicycle",
                 "child riding tricycle", "vintage kids tricycle", "tricycle bike with basket", "red tricycle",
                 "senior riding adult tricycle", "metal tricycle for kids"],
    "road": ["road bike", "road bicycle side view", "carbon road bike parked", "racing bicycle drop handlebars",
             "steel road bike", "road bike leaning against wall"],
    "mountain": ["mountain bike", "hardtail mountain bike", "full suspension mountain bike", "mountain bike parked trail",
                 "mountain bike side view"],
    "cruiser": ["beach cruiser bicycle", "cruiser bike side view", "beach cruiser bike parked", "womens beach cruiser bike",
                "mens cruiser bicycle", "vintage cruiser bicycle"],
}
URL_LOG = []  # (tmp filename, image url) filled by the downloader hook


class LoggingDownloader(ImageDownloader):
    def process_meta(self, task):
        if task.get("success", True) and task.get("filename"):
            URL_LOG.append((task["filename"], task["file_url"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(QUERIES))
    ap.add_argument("--per-query", type=int, default=80)
    args = ap.parse_args()

    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    seen_sha = {r["sha1"] for r in rows}
    seen_url = {r["image_url"] for r in rows}
    for key in args.classes:
        label = key.split("_v")[0]  # e.g. "bmx_v2" -> label "bmx"
        added = 0
        for q in QUERIES[key]:
            tmp = Path(tempfile.mkdtemp(prefix="bing_"))
            URL_LOG.clear()
            crawler = BingImageCrawler(downloader_cls=LoggingDownloader, storage={"root_dir": str(tmp)},
                                       downloader_threads=4, log_level=logging.ERROR)
            # icrawler puts the keyword into the URL unencoded, so only the first word was searched.
            crawler.crawl(keyword=quote_plus(q), max_num=args.per_query, filters={"type": "photo"})
            for fname, url in URL_LOG:
                path = tmp / fname
                if url in seen_url or not path.exists():
                    continue
                content = path.read_bytes()
                sha = hashlib.sha1(content).hexdigest()
                if sha in seen_sha:
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    if min(img.size) < 128:
                        continue
                    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
                    out = f"{label}/{label}_bing_{sha[:12]}.jpg"
                    img.save(RAW_DIR / out, "JPEG", quality=90)
                except Exception:
                    continue
                seen_sha.add(sha)
                seen_url.add(url)
                rows.append(dict(file=out, label=label, source="bing", source_id=q, page_url="", image_url=url,
                                 license="unknown (web search; evaluation only, not redistributed)",
                                 author=urlparse(url).netloc.removeprefix("www."), sha1=sha))
                added += 1
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  {label} / '{q}': total added {added}")
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{label}: +{added}")


if __name__ == "__main__":
    main()
