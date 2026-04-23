import requests
import zipfile
import json
import random
import csv
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

# CONFIGURATION
ANNOTATIONS_URLS = [
    "https://cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]
ANNOTATIONS_PATH_IN_ZIP = "annotations/instances_val2017.json"

TARGET_DIR = Path("images")
NUM_IMAGES = 1024
SEED = 1234

MANIFEST_JSON = Path("images_manifest.json")
MANIFEST_CSV = Path("images_manifest.csv")


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def download_coco_samples():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Downloading COCO metadata (~240MB)...")
    zip_bytes = None
    last_err = None
    for url in ANNOTATIONS_URLS:
        try:
            print(f"   Trying: {url}")
            zip_bytes = fetch_bytes(url, timeout=120)
            annotations_url_used = url
            break
        except Exception as e:
            print(f"   Failed: {e}")
            last_err = e

    if zip_bytes is None:
        raise RuntimeError(f"Could not download annotations from any URL. Last error: {last_err}")

    print("2. Extracting image URLs from metadata...")
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        with z.open(ANNOTATIONS_PATH_IN_ZIP) as f:
            data = json.load(f)

    all_images = data["images"]
    print(f"   Found {len(all_images)} valid images in validation set.")

    if NUM_IMAGES > len(all_images):
        raise ValueError(f"NUM_IMAGES ({NUM_IMAGES}) > available images ({len(all_images)}).")

    random.seed(SEED)
    selected_images = random.sample(all_images, NUM_IMAGES)

    manifest_records = []
    for img in selected_images:
        manifest_records.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "coco_url": img["coco_url"],   # often http://images.cocodataset.org/...
            "width": img.get("width"),
            "height": img.get("height"),
            "license": img.get("license"),
            "flickr_url": img.get("flickr_url"),
        })
    manifest_records.sort(key=lambda x: x["id"])

    print(f"3. Downloading {NUM_IMAGES} unique images to '{TARGET_DIR}' (seed={SEED})...")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(manifest_records, start=1):
        save_path = TARGET_DIR / rec["file_name"]

        if save_path.exists():
            print(f"   [{i}/{NUM_IMAGES}] Already exists: {rec['file_name']} (skipping)")
            skipped += 1
            continue

        try:
            r = requests.get(rec["coco_url"], timeout=30)
            r.raise_for_status()

            # Light sanity check if server returns content-type
            ctype = r.headers.get("Content-Type", "")
            if ctype and not ctype.startswith("image/"):
                raise ValueError(f"Unexpected Content-Type: {ctype}")

            save_path.write_bytes(r.content)
            print(f"   [{i}/{NUM_IMAGES}] Downloaded {rec['file_name']}")
            downloaded += 1
        except Exception as e:
            print(f"   [ERROR] Failed to download {rec['file_name']}: {e}")
            failed += 1

    manifest = {
        "dataset": "COCO 2017 val",
        "annotations": ANNOTATIONS_PATH_IN_ZIP,
        "annotations_urls_tried": ANNOTATIONS_URLS,
        "annotations_url_used": annotations_url_used,
        "num_images_requested": NUM_IMAGES,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "images": manifest_records,
        "download_summary": {
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
        }
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    print(f"\n4. Wrote manifest JSON: {MANIFEST_JSON.resolve()}")

    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "file_name", "coco_url", "width", "height", "license", "flickr_url"]
        )
        writer.writeheader()
        for rec in manifest_records:
            writer.writerow(rec)

    print(f"5. Wrote manifest CSV:  {MANIFEST_CSV.resolve()}")

    print("\n✅ Done.")


if __name__ == "__main__":
    download_coco_samples()
