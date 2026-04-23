import os
import json
import zipfile
import random
import numpy as np
import requests
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from datetime import datetime, timezone

# CONFIGURATION
TASK_DIR = Path("coco_segmentation")
IMAGE_DIR = TASK_DIR / "images"
MASK_DIR = TASK_DIR / "masks"
NUM_IMAGES = 512
SEED = 1234

MANIFEST_JSON = TASK_DIR / "manifest.json"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP = TASK_DIR / "annotations_trainval2017.zip"
ANNOTATIONS_FILE = TASK_DIR / "annotations" / "instances_val2017.json"

# torchvision DeepLabV3 uses COCO_WITH_VOC_LABELS. 
# We must map the 80 COCO categories down to the 20 VOC IDs the model predicts.
COCO_TO_VOC = {
    5: 1,   # airplane -> aeroplane
    2: 2,   # bicycle -> bicycle
    16: 3,  # bird -> bird
    9: 4,   # boat -> boat
    44: 5,  # bottle -> bottle
    6: 6,   # bus -> bus
    3: 7,   # car -> car
    17: 8,  # cat -> cat
    62: 9,  # chair -> chair
    21: 10, # cow -> cow
    67: 11, # dining table -> diningtable
    18: 12, # dog -> dog
    19: 13, # horse -> horse
    4: 14,  # motorcycle -> motorbike
    1: 15,  # person -> person
    64: 16, # potted plant -> pottedplant
    20: 17, # sheep -> sheep
    63: 18, # couch -> sofa
    7: 19,  # train -> train
    72: 20  # tv -> tvmonitor
}

def download_file(url, filepath):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def prepare_coco_segmentation():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and extract official annotations (~241MB)
    if not ANNOTATIONS_FILE.exists():
        print("1. Downloading official COCO annotations (~241MB)...")
        download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP)
        print("   Extracting annotations...")
        with zipfile.ZipFile(ANNOTATIONS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(TASK_DIR)
        ANNOTATIONS_ZIP.unlink() # Cleanup zip file to save space
    else:
        print("1. COCO annotations already found locally.")

    print("2. Loading COCO API...")
    coco = COCO(ANNOTATIONS_FILE)

    print("3. Filtering images for VOC classes...")
    # Get all images that contain at least one of our 20 target classes
    valid_img_ids = set()
    for coco_id in COCO_TO_VOC.keys():
        valid_img_ids.update(coco.getImgIds(catIds=[coco_id]))
    
    valid_img_ids = list(valid_img_ids)
    valid_img_ids.sort() # Ensure stable seed across runs
    random.seed(SEED)
    selected_img_ids = random.sample(valid_img_ids, min(NUM_IMAGES, len(valid_img_ids)))

    print(f"4. Downloading {len(selected_img_ids)} images and rendering masks...")
    manifest_records = []
    
    for i, img_id in enumerate(selected_img_ids, start=1):
        img_info = coco.loadImgs(img_id)[0]
        
        # Download Image directly from COCO
        img_filename = f"coco_seg_val_{i:04d}.jpg"
        img_path = IMAGE_DIR / img_filename
        if not img_path.exists():
            download_file(img_info['coco_url'], img_path)
            
        # Standardize to RGB
        img_pil = Image.open(img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
            img_pil.save(img_path)

        # Generate Blank Mask (0 = Background)
        mask_array = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # SORT BY AREA DESCENDING: Draw large objects first, small objects on top
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in COCO_TO_VOC:
                voc_id = COCO_TO_VOC[cat_id]
                try:
                    # pycocotools decodes the polygon into a 2D binary matrix
                    binary_mask = coco.annToMask(ann)
                    # Apply the VOC class ID to the mask array where the binary mask is 1
                    mask_array[binary_mask == 1] = voc_id
                except Exception:
                    continue # Skip rarely corrupted RLEs
        
        # Save Mask as PNG (PNG is lossless; do not use JPEG for class ID arrays)
        mask_filename = f"coco_seg_val_{i:04d}_mask.png"
        mask_path = MASK_DIR / mask_filename
        Image.fromarray(mask_array).save(mask_path)
        
        manifest_records.append({
            "id": i,
            "original_coco_id": img_id,
            "file_name": img_filename,
            "mask_name": mask_filename,
            "width": img_info['width'],
            "height": img_info['height']
        })
        
        if i % 32 == 0 or i == len(selected_img_ids):
            print(f"   [{i}/{len(selected_img_ids)}] Processed...")

    print("\n5. Writing manifest...")
    manifest = {
        "dataset": "COCO 2017 Validation (VOC Classes Subset)",
        "num_images": len(selected_img_ids),
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data": manifest_records
    }
    
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Done! Semantic segmentation data organized in '{TASK_DIR}/'")

if __name__ == "__main__":
    prepare_coco_segmentation()