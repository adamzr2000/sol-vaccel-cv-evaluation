import json
import csv
from pathlib import Path
from datasets import load_dataset
from datetime import datetime, timezone
import os

# CONFIGURATION
TASK_DIR = Path("imagenet")
TARGET_DIR = TASK_DIR / "images"
NUM_IMAGES = 512
SEED = 1234

MANIFEST_JSON = TASK_DIR / "manifest.json"
MANIFEST_CSV = TASK_DIR / "manifest.csv"

def download_imagenet_samples():
    # Create the nested directory structure
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"1. Connecting to Hugging Face to stream ImageNet-1K (ILSVRC) validation set...")
    try:
        dataset = load_dataset(
            "ILSVRC/imagenet-1k", 
            split="validation", 
            streaming=True, 
            token=True,
        )

        # Grab class names from the features metadata
        class_names = dataset.info.features['label'].names
    except Exception as e:
        print(f"❌ Error connecting to ImageNet: {e}")
        print("Check your login status: `huggingface-cli login` and dataset access.")
        return

    print(f"2. Shuffling dataset and selecting {NUM_IMAGES} images (seed={SEED})...")
    shuffled_dataset = dataset.shuffle(seed=SEED, buffer_size=5000)
    selected_samples = shuffled_dataset.take(NUM_IMAGES)

    manifest_records = []
    downloaded = 0
    
    print(f"3. Downloading and saving images to '{TARGET_DIR}'...")
    
    for i, sample in enumerate(selected_samples, start=1):
        img = sample['image']
        label_id = sample['label']
        label_text = class_names[label_id]
        
        # Filename includes sequence and label for easy manual inspection
        file_name = f"imagenet_val_{i:04d}_label_{label_id}.jpg"
        save_path = TARGET_DIR / file_name
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(save_path, "JPEG")
        
        manifest_records.append({
            "id": i,
            "file_name": file_name,
            "ground_truth_id": label_id,
            "ground_truth_label": label_text,
            "width": img.width,
            "height": img.height
        })
        
        if i % 50 == 0 or i == NUM_IMAGES:
            print(f"   [{i}/{NUM_IMAGES}] Processed...")
        downloaded += 1

    print("\n4. Writing manifests...")
    
    manifest = {
        "dataset": "ImageNet-1K (ILSVRC-2012) validation",
        "num_images_requested": NUM_IMAGES,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "images": manifest_records,
        "download_summary": {"downloaded": downloaded}
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["id", "file_name", "ground_truth_id", "ground_truth_label", "width", "height"]
        )
        writer.writeheader()
        for rec in manifest_records:
            writer.writerow(rec)

    print(f"✅ Done. Data organized in '{TASK_DIR}/'")
    print(f"   - Images: {TARGET_DIR}/")
    print(f"   - JSON Manifest: {MANIFEST_JSON}")
    print(f"   - CSV Manifest: {MANIFEST_CSV}")

    os._exit(0)

if __name__ == "__main__":
    download_imagenet_samples()