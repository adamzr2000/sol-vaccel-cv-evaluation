import json
import csv
import shutil
from pathlib import Path
from datasets import load_dataset
from datetime import datetime, timezone

# CONFIGURATION
TASK_DIR = Path("kinetics")
TARGET_DIR = TASK_DIR / "videos"
NUM_VIDEOS = 64
SEED = 1234

MANIFEST_JSON = TASK_DIR / "manifest.json"
MANIFEST_CSV = TASK_DIR / "manifest.csv"

# Note: You may need to change this if you are using a specific Kinetics subset/repo 
# (e.g., "kashif/kinetics-400", "AlexZigma/kinetics-400", or a Kinetics-700 variant)
HF_DATASET_NAME = "nateraw/kinetics" 

def download_kinetics_samples():
    # Create the nested directory structure
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. Connecting to Hugging Face to stream Kinetics validation set...")
    try:
        dataset = load_dataset(
            "parquet",
            data_files="hf://datasets/nateraw/kinetics@~parquet/default/validation/0000.parquet",
            split="train", # The parquet loader treats the file itself as the base split
            streaming=True,
            token=True,
        )

        # Grab class names from the features metadata
        class_names = dataset.info.features['label'].names
    except Exception as e:
        print(f"❌ Error connecting to Kinetics dataset: {e}")
        print("Check your login status: `huggingface-cli login` and dataset access/name.")
        return

    print(f"2. Shuffling dataset and selecting {NUM_VIDEOS} videos (seed={SEED})...")
    shuffled_dataset = dataset.shuffle(seed=SEED, buffer_size=1000)
    selected_samples = shuffled_dataset.take(NUM_VIDEOS)

    manifest_records = []
    downloaded = 0

    print(f"3. Downloading and saving videos to '{TARGET_DIR}'...")

    for i, sample in enumerate(selected_samples, start=1):
        video_data = sample['video']
        label_id = sample['label']
        label_text = class_names[label_id]

        # Filename includes sequence and label for easy manual inspection
        file_name = f"kinetics_val_{i:04d}_label_{label_id}.mp4"
        save_path = TARGET_DIR / file_name

        try:
            # 1. Handle raw binary bytes (Parquet format)
            if isinstance(video_data, bytes):
                with open(save_path, "wb") as f:
                    f.write(video_data)
            
            # 2. Handle HF dictionary format with bytes
            elif isinstance(video_data, dict) and 'bytes' in video_data and video_data['bytes'] is not None:
                with open(save_path, "wb") as f:
                    f.write(video_data['bytes'])
            
            # 3. Handle HF dictionary format with a local cache path
            elif isinstance(video_data, dict) and 'path' in video_data and video_data['path'] is not None:
                shutil.copy2(video_data['path'], save_path)
            
            # 4. Handle direct string path
            elif isinstance(video_data, str):
                shutil.copy2(video_data, save_path)
            
            else:
                print(f"⚠️ Unrecognized video format for sample {i} (Type: {type(video_data)}), skipping.")
                continue
                
        except Exception as e:
            print(f"⚠️ Failed to save video {i}: {e}")
            continue

        manifest_records.append({
            "id": i,
            "file_name": file_name,
            "ground_truth_id": label_id,
            "ground_truth_label": label_text
        })

        if i % 8 == 0 or i == NUM_VIDEOS:
            print(f"   [{i}/{NUM_VIDEOS}] Processed...")
        downloaded += 1

    print("\n4. Writing manifests...")

    manifest = {
        "dataset": HF_DATASET_NAME,
        "num_videos_requested": NUM_VIDEOS,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "videos": manifest_records,
        "download_summary": {"downloaded": downloaded}
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "file_name", "ground_truth_id", "ground_truth_label"]
        )
        writer.writeheader()
        for rec in manifest_records:
            writer.writerow(rec)

    print(f"✅ Done. Data organized in '{TASK_DIR}/'")
    print(f"   - Videos: {TARGET_DIR}/")
    print(f"   - JSON Manifest: {MANIFEST_JSON}")
    print(f"   - CSV Manifest: {MANIFEST_CSV}")

if __name__ == "__main__":
    download_kinetics_samples()