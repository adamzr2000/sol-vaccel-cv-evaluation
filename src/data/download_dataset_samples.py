#!/usr/bin/env python3
"""
Unified dataset downloader for benchmark evaluation.
Prepares 1,024 sample images/videos from each relevant dataset:
  - ImageNet-1k val: for image classification models (requires local copy)
  - UCF101: for video classification models (requires local copy)
  - COCO val: for object detection & segmentation models (downloads from COCO)

Each dataset is saved to its own directory with a corresponding labels CSV.
"""

import os
import sys
import json
import csv
import argparse
import shutil
import random
from pathlib import Path
from datetime import datetime, timezone


# =========================================================
# CONFIGURATION
# =========================================================
CONFIG = {
    "imagenet": {
        "output_dir": Path("imagenet_val_1k"),
        "num_samples": 1024,
        "seed": 42,
    },
    "ucf101": {
        "output_dir": Path("ucf101_1k"),
        "num_samples": 1024,
        "seed": 42,
    },
    "coco_detection": {
        "output_dir": Path("coco_detection_1k"),
        "num_samples": 1024,
        "seed": 42,
    },
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_imagenet_samples():
    """
    Create ImageNet labels for local ImageNet dataset (requires user to have downloaded ImageNet).
    Copies images to output dir and creates labels.csv.
    """
    config = CONFIG["imagenet"]
    output_dir = config["output_dir"]
    num_samples = config["num_samples"]
    seed = config["seed"]

    ensure_dir(output_dir)

    print(f"📥 Setting up ImageNet-1k ({num_samples} samples)...")
    print(f"   Output: {output_dir.resolve()}")

    # Check if user has local ImageNet in a standard location
    imagenet_root = Path.home() / "data" / "imagenet" / "val"
    if not imagenet_root.exists():
        imagenet_root = Path("/data/imagenet/val")
    if not imagenet_root.exists():
        imagenet_root = Path("/mnt/imagenet/val")

    if not imagenet_root.exists():
        print(f"""
❌ ImageNet validation set not found in standard locations:
   - ~/data/imagenet/val
   - /data/imagenet/val
   - /mnt/imagenet/val

📋 To use ImageNet:
   1. Download ImageNet-1k from https://www.image-net.org/download.php
   2. Extract to one of the paths above
   3. Re-run this script

Skipping ImageNet for now.
""")
        return

    print(f"   Found ImageNet root: {imagenet_root}")

    # Scan for class directories
    class_dirs = sorted([d for d in imagenet_root.iterdir() if d.is_dir()])
    if not class_dirs:
        print("❌ No class directories found in ImageNet root")
        return

    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    print(f"   Found {len(class_dirs)} classes, sampling {num_samples} images...")

    # Randomly sample images across classes
    all_images = []
    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg"))
        for img_file in image_files:
            all_images.append((img_file, class_id, class_name))

    if len(all_images) < num_samples:
        print(f"❌ Not enough images: {len(all_images)} < {num_samples}")
        return

    random.seed(seed)
    sampled = random.sample(all_images, num_samples)

    labels_records = []
    for i, (src_path, class_id, class_name) in enumerate(sampled):
        file_name = f"imagenet_{i:06d}.jpg"
        dst_path = images_dir / file_name

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"   ⚠️  Failed to copy {src_path}: {e}")
            continue

        labels_records.append({
            "filename": file_name,
            "class_id": class_id,
            "class_name": class_name,
        })

        if (i + 1) % 100 == 0:
            print(f"   [{i + 1}/{num_samples}] Copied...")

    # Save labels CSV
    labels_csv = output_dir / "labels.csv"
    with labels_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class_id", "class_name"])
        writer.writeheader()
        writer.writerows(labels_records)

    # Save manifest JSON
    manifest = {
        "dataset": "ImageNet-1k validation",
        "num_samples": len(labels_records),
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "images_dir": str(images_dir.relative_to(Path.cwd())),
        "labels_csv": str(labels_csv.relative_to(Path.cwd())),
        "total_classes": len(set(r["class_id"] for r in labels_records)),
    }

    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"✅ ImageNet-1k: {len(labels_records)} images saved to {images_dir}")
    print(f"   Labels: {labels_csv}")
    print(f"   Manifest: {manifest_json}")



def save_ucf101_samples():
    """Prepare UCF101 dataset from local copy and save with labels."""
    config = CONFIG["ucf101"]
    output_dir = config["output_dir"]
    num_samples = config["num_samples"]
    seed = config["seed"]

    ensure_dir(output_dir)
    videos_dir = output_dir / "videos"
    ensure_dir(videos_dir)

    print(f"📥 Setting up UCF101 ({num_samples} videos)...")
    print(f"   Output: {output_dir.resolve()}")

    # Check if UCF101 already exists locally
    ucf101_root = Path.home() / "data" / "UCF101"
    if not ucf101_root.exists():
        ucf101_root = Path("/data/UCF101")
    if not ucf101_root.exists():
        ucf101_root = Path("/mnt/UCF101")

    if not ucf101_root.exists():
        print(f"""
❌ UC101 dataset not found in standard locations:
   - ~/data/UCF101
   - /data/UCF101
   - /mnt/UCF101

📋 To use UCF101:
   1. Download from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar (or .tar.gz)
   2. Extract to one of the paths above
   3. Re-run this script

Skipping UCF101 for now.
""")
        return

    print(f"   Found UCF101 root: {ucf101_root}")

    # Scan for action directories and videos
    action_dirs = sorted([d for d in ucf101_root.iterdir() if d.is_dir()])
    if not action_dirs:
        print("❌ No action directories found in UCF101 root")
        return

    print(f"   Found {len(action_dirs)} action classes, sampling {num_samples} videos...")

    # Randomly sample videos across actions
    all_videos = []
    for action_id, action_dir in enumerate(action_dirs):
        action_name = action_dir.name
        video_files = sorted(action_dir.glob("*.avi")) + sorted(action_dir.glob("*.mp4"))
        for vid_file in video_files:
            all_videos.append((vid_file, action_id, action_name))

    if len(all_videos) < num_samples:
        print(f"❌ Not enough videos: {len(all_videos)} < {num_samples}")
        num_samples = len(all_videos)

    random.seed(seed)
    sampled = random.sample(all_videos, num_samples)

    labels_records = []
    for i, (src_path, action_id, action_name) in enumerate(sampled):
        file_name = f"ucf101_{i:06d}.avi"
        dst_path = videos_dir / file_name

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"   ⚠️  Failed to copy {src_path}: {e}")
            continue

        labels_records.append({
            "filename": file_name,
            "action_id": action_id,
            "action_name": action_name,
        })

        if (i + 1) % 100 == 0:
            print(f"   [{i + 1}/{num_samples}] Copied...")

    # Save labels CSV
    labels_csv = output_dir / "labels.csv"
    with labels_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "action_id", "action_name"])
        writer.writeheader()
        writer.writerows(labels_records)

    # Save manifest JSON
    manifest = {
        "dataset": "UCF101",
        "num_samples": len(labels_records),
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "videos_dir": str(videos_dir.relative_to(Path.cwd())),
        "labels_csv": str(labels_csv.relative_to(Path.cwd())),
        "total_actions": len(set(r["action_id"] for r in labels_records)),
    }

    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"✅ UCF101: {len(labels_records)} videos saved to {videos_dir}")
    print(f"   Labels: {labels_csv}")
    print(f"   Manifest: {manifest_json}")


def save_coco_samples():
    """Download COCO val 2017 images and save with labels."""
    import requests
    import zipfile
    from io import BytesIO

    config = CONFIG["coco_detection"]
    output_dir = config["output_dir"]
    num_samples = config["num_samples"]
    seed = config["seed"]

    ensure_dir(output_dir)
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    print(f"📥 Downloading COCO val 2017 subset ({num_samples} images)...")
    print(f"   Output: {output_dir.resolve()}")

    # Download COCO metadata
    ANNOTATIONS_URLS = [
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "https://dl.fbaipublicfiles.com/datasets/coco/coco2017/annotations_trainval2017.zip",
    ]
    ANNOTATIONS_PATH_IN_ZIP = "annotations/instances_val2017.json"

    print("   Downloading COCO metadata (~240MB)...")
    zip_bytes = None
    last_err = None
    for url in ANNOTATIONS_URLS:
        try:
            print(f"      Trying: {url}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            zip_bytes = resp.content
            break
        except Exception as e:
            print(f"      Failed: {e}")
            last_err = e

    if zip_bytes is None:
        print(f"❌ Could not download COCO annotations. Last error: {last_err}")
        return

    print("   Extracting image URLs from metadata...")
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        with z.open(ANNOTATIONS_PATH_IN_ZIP) as f:
            data = json.load(f)

    all_images = data["images"]
    print(f"   Found {len(all_images)} images in COCO val 2017")

    if num_samples > len(all_images):
        num_samples = len(all_images)

    random.seed(seed)
    selected_images = random.sample(all_images, num_samples)

    print(f"   Downloading {num_samples} images...")
    detection_labels = []
    downloaded = 0

    for i, img_rec in enumerate(selected_images):
        file_name = f"coco_{i:06d}.jpg"
        image_path = images_dir / file_name

        try:
            resp = requests.get(img_rec["coco_url"], timeout=30)
            resp.raise_for_status()
            image_path.write_bytes(resp.content)
            downloaded += 1

            detection_labels.append({
                "filename": file_name,
                "image_id": img_rec["id"],
                "coco_url": img_rec["coco_url"],
            })

            if (i + 1) % 100 == 0:
                print(f"   [{i + 1}/{num_samples}] Downloaded...")
        except Exception as e:
            print(f"   ⚠️  Failed to download image {i}: {e}")

    # Save labels CSV
    labels_csv = output_dir / "labels_detection.csv"
    with labels_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "image_id", "coco_url"]
        )
        writer.writeheader()
        writer.writerows(detection_labels)

    # Save manifest JSON
    manifest = {
        "dataset": "COCO 2017 validation",
        "num_samples": downloaded,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "images_dir": str(images_dir.relative_to(Path.cwd())),
        "labels_detection_csv": str(labels_csv.relative_to(Path.cwd())),
    }

    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"✅ COCO val: {downloaded} images saved to {images_dir}")
    print(f"   Labels: {labels_csv}")
    print(f"   Manifest: {manifest_json}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download dataset samples for benchmark evaluation."
    )
    parser.add_argument(
        "--dataset",
        choices=["imagenet", "ucf101", "coco", "all"],
        default="all",
        help="Which dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of samples per dataset (default: 1024)",
    )

    args = parser.parse_args()

    # Update config with user overrides
    for dataset_key in CONFIG:
        CONFIG[dataset_key]["num_samples"] = args.num_samples

    print("=" * 70)
    print("DATASET DOWNLOADER FOR BENCHMARK EVALUATION")
    print("=" * 70)

    if args.dataset in ("imagenet", "all"):
        save_imagenet_samples()
        print()

    if args.dataset in ("ucf101", "all"):
        save_ucf101_samples()
        print()

    if args.dataset in ("coco", "all"):
        save_coco_samples()
        print()

    print("=" * 70)
    print("✅ All downloads complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
