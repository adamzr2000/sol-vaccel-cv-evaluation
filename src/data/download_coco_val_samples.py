import json
from pathlib import Path
from datasets import load_dataset
from datetime import datetime, timezone

# CONFIGURATION
TASK_DIR = Path("coco_detection")
IMAGE_DIR = TASK_DIR / "images"
NUM_IMAGES = 512
SEED = 1234

MANIFEST_JSON = TASK_DIR / "manifest.json"

def download_coco_detection_samples():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Connecting to Hugging Face to stream COCO validation set...")
    try:
        # 'detection-datasets/coco' is optimized for bounding box detection
        dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"2. Shuffling and selecting {NUM_IMAGES} samples for Object Detection...")
    selected_samples = dataset.shuffle(seed=SEED, buffer_size=1000).take(NUM_IMAGES)

    manifest_records = []
    
    print(f"3. Saving images and extracting bounding boxes to '{TASK_DIR}'...")
    
    for i, sample in enumerate(selected_samples, start=1):
        # 1. Save the Image
        img = sample['image']
        img_filename = f"coco_val_{i:04d}.jpg"
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(IMAGE_DIR / img_filename)
        
        # 2. Extract Object Detection Ground Truth
        objects = sample['objects']
        
        # 3. Add to manifest
        manifest_records.append({
            "id": i,
            "file_name": img_filename,
            "width": img.width,
            "height": img.height,
            "objects": {
                "bboxes": objects['bbox'],       # Format: [x_min, y_min, width, height]
                "categories": objects['category'] # COCO class IDs
            }
        })
        
        if i % 64 == 0 or i == NUM_IMAGES:
            print(f"   [{i}/{NUM_IMAGES}] Processed...")

    print("\n4. Writing manifest...")
    manifest = {
        "dataset": "COCO 2017 Validation Subset (Detection)",
        "num_images": NUM_IMAGES,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data": manifest_records
    }
    
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Done! Detection data organized in '{TASK_DIR}/'")

if __name__ == "__main__":
    download_coco_detection_samples()