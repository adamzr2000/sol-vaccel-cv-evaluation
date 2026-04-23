import csv
import json
import random
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# --------------------
# CONFIG
# --------------------
K400_VAL_PATHS_TXT_URLS = [
    "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt",
]

# Official Kinetics-400 annotations (used by torchvision.datasets.Kinetics too)
K400_VAL_ANNOTATIONS_URLS = [
    "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
]

TARGET_DIR = Path("videos")
NUM_VIDEOS = 64
SEED = 1234

TMP_DIR = Path("_tmp_kinetics_k400_val")

MANIFEST_JSON = Path("videos_manifest.json")
MANIFEST_CSV = Path("videos_manifest.csv")

HTTP_TIMEOUT = 120
RETRY_SHARDS = 2


# --------------------
# Helpers
# --------------------
def fetch_text(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def download_file_stream(url: str, dst: Path, timeout: int = HTTP_TIMEOUT) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def parse_paths_txt(text: str) -> List[str]:
    urls: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    # Deduplicate, keep order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def load_kinetics_val_annotations() -> Tuple[Dict[str, str], str]:
    """
    Returns:
      - mapping: clip_filename (e.g. ytid_000065_000075.mp4) -> label string
      - url_used
    """
    last_err = None
    for url in K400_VAL_ANNOTATIONS_URLS:
        try:
            print("1b) Downloading Kinetics-400 val annotations (val.csv)...")
            print(f"    Trying: {url}")
            txt = fetch_text(url, timeout=HTTP_TIMEOUT)
            mapping: Dict[str, str] = {}

            reader = csv.DictReader(StringIO(txt))
            # torchvision expects columns: youtube_id, time_start, time_end, label, ...
            for row in reader:
                ytid = row["youtube_id"]
                start = int(row["time_start"])
                end = int(row["time_end"])
                label = row.get("label") or row.get("classname") or row.get("action")  # defensive
                if not label:
                    continue
                fname = f"{ytid}_{start:06d}_{end:06d}.mp4"
                mapping[fname] = label

            if not mapping:
                raise RuntimeError("Parsed val.csv but got 0 rows/mappings (format changed?).")

            print(f"    Loaded {len(mapping)} annotation rows.")
            return mapping, url
        except Exception as e:
            print(f"    Failed: {e}")
            last_err = e

    raise RuntimeError(f"Could not download/parse Kinetics annotations. Last error: {last_err}")


def build_label_to_index_from_torchvision() -> Optional[Dict[str, int]]:
    """
    Optional: map label string -> class index matching torchvision pretrained weights.
    If torchvision isn't available or weights metadata format differs, return None.
    """
    try:
        # Either R3D_18_Weights or MC3_18_Weights should have the same Kinetics-400 categories list.
        from torchvision.models.video import R3D_18_Weights  # type: ignore

        cats = R3D_18_Weights.DEFAULT.meta.get("categories")
        if not cats:
            return None
        return {name: i for i, name in enumerate(cats)}
    except Exception:
        return None


@dataclass
class ClipRecord:
    file_name: str                 # actual saved file name (may include collision suffix)
    clip_key: str                  # original kinetics key filename ytid_000000_000000.mp4
    label: Optional[str]
    label_index: Optional[int]
    source_shard_url: str
    source_member_path: str


# --------------------
# Main logic
# --------------------
def extract_mp4s_from_tar(
    tar_path: Path,
    target_dir: Path,
    need: int,
    rng: random.Random,
    shard_url: str,
    ann_map: Dict[str, str],
    label_to_index: Optional[Dict[str, int]],
) -> Tuple[List[ClipRecord], int, int]:
    """
    Extract up to `need` mp4 members from `tar_path` into `target_dir`.
    Returns (records, extracted_count, missing_label_count).
    """
    records: List[ClipRecord] = []
    extracted = 0
    missing_labels = 0

    with tarfile.open(tar_path, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".mp4")]
        if not members:
            return records, 0, 0

        members_sorted = sorted(members, key=lambda m: m.name)
        rng.shuffle(members_sorted)

        for m in members_sorted:
            if extracted >= need:
                break

            base = Path(m.name).name                # e.g. ytid_000065_000075.mp4
            clip_key = base                         # keep original key for annotation lookup

            label = ann_map.get(clip_key)
            if label is None:
                missing_labels += 1

            label_idx = None
            if label is not None and label_to_index is not None:
                label_idx = label_to_index.get(label)

            out_name = base
            out_path = target_dir / out_name

            # Handle collisions (rare, but possible)
            if out_path.exists():
                stem = out_path.stem
                suffix = out_path.suffix
                tag = str(abs(hash(m.name)) % 1_000_000)
                out_name = f"{stem}__{tag}{suffix}"
                out_path = target_dir / out_name

            try:
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                out_path.write_bytes(fobj.read())

                records.append(
                    ClipRecord(
                        file_name=out_name,
                        clip_key=clip_key,
                        label=label,
                        label_index=label_idx,
                        source_shard_url=shard_url,
                        source_member_path=m.name,
                    )
                )
                extracted += 1
            except Exception as e:
                print(f"      [WARN] Failed extracting {m.name}: {e}")

    return records, extracted, missing_labels


def download_kinetics_val_video_samples():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotations first so labels are correct
    ann_map, ann_url_used = load_kinetics_val_annotations()
    label_to_index = build_label_to_index_from_torchvision()
    if label_to_index is not None:
        print("1c) Loaded torchvision categories -> will write label_index too.")
    else:
        print("1c) Could not load torchvision categories (label_index will be null).")

    print("2) Fetching Kinetics-400 val shard list...")
    paths_text = None
    last_err = None
    for url in K400_VAL_PATHS_TXT_URLS:
        try:
            print(f"   Trying: {url}")
            paths_text = fetch_text(url)
            paths_url_used = url
            break
        except Exception as e:
            print(f"   Failed: {e}")
            last_err = e

    if paths_text is None:
        raise RuntimeError(f"Could not fetch shard list. Last error: {last_err}")

    shard_urls = parse_paths_txt(paths_text)
    if not shard_urls:
        raise RuntimeError("Shard list parsed empty. The path file format may have changed.")

    print(f"   Found {len(shard_urls)} shard URLs.")

    rng = random.Random(SEED)
    rng.shuffle(shard_urls)

    print(f"3) Downloading/extracting until we have {NUM_VIDEOS} clips (seed={SEED})...")
    total_records: List[ClipRecord] = []
    downloaded_shards = 0
    failed_shards = 0
    missing_labels_total = 0

    need = NUM_VIDEOS

    for idx, shard_url in enumerate(shard_urls, start=1):
        if need <= 0:
            break

        shard_name = Path(shard_url).name
        tar_path = TMP_DIR / shard_name

        print(f"   [{idx}/{len(shard_urls)}] Shard: {shard_name} (need {need} more)")

        ok = False
        for attempt in range(1, RETRY_SHARDS + 1):
            try:
                if not tar_path.exists():
                    print(f"      Downloading (attempt {attempt})...")
                    download_file_stream(shard_url, tar_path)
                ok = True
                break
            except Exception as e:
                print(f"      [WARN] Download failed: {e}")
                if tar_path.exists():
                    try:
                        tar_path.unlink()
                    except Exception:
                        pass

        if not ok:
            failed_shards += 1
            continue

        downloaded_shards += 1

        try:
            recs, extracted, missing_labels = extract_mp4s_from_tar(
                tar_path=tar_path,
                target_dir=TARGET_DIR,
                need=need,
                rng=rng,
                shard_url=shard_url,
                ann_map=ann_map,
                label_to_index=label_to_index,
            )
            total_records.extend(recs)
            need -= extracted
            missing_labels_total += missing_labels
            print(f"      Extracted {extracted} clips from shard. (missing labels: {missing_labels})")
        except Exception as e:
            print(f"      [WARN] Extraction failed: {e}")
            failed_shards += 1

        try:
            tar_path.unlink()
        except Exception:
            pass

    downloaded = len(total_records)
    if downloaded < NUM_VIDEOS:
        print(f"\n[WARN] Only got {downloaded}/{NUM_VIDEOS} clips.")

    total_records.sort(key=lambda r: r.file_name)

    manifest = {
        "dataset": "Kinetics-400 (CVDF hosted) val sample",
        "paths_txt_urls_tried": K400_VAL_PATHS_TXT_URLS,
        "paths_txt_url_used": paths_url_used,
        "annotations_urls_tried": K400_VAL_ANNOTATIONS_URLS,
        "annotations_url_used": ann_url_used,
        "num_videos_requested": NUM_VIDEOS,
        "num_videos_downloaded": downloaded,
        "seed": SEED,
        "target_dir": str(TARGET_DIR),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "download_summary": {
            "downloaded_shards": downloaded_shards,
            "failed_shards": failed_shards,
            "missing_labels": missing_labels_total,
        },
        "videos": [
            {
                "file_name": r.file_name,
                "clip_key": r.clip_key,
                "label": r.label,
                "label_index": r.label_index,
                "source_shard_url": r.source_shard_url,
                "source_member_path": r.source_member_path,
            }
            for r in total_records
        ],
    }

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    print(f"\n4) Wrote manifest JSON: {MANIFEST_JSON.resolve()}")

    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "clip_key", "label", "label_index", "source_shard_url", "source_member_path"],
        )
        writer.writeheader()
        for r in total_records:
            writer.writerow(
                {
                    "file_name": r.file_name,
                    "clip_key": r.clip_key,
                    "label": r.label,
                    "label_index": r.label_index,
                    "source_shard_url": r.source_shard_url,
                    "source_member_path": r.source_member_path,
                }
            )

    print(f"5) Wrote manifest CSV:  {MANIFEST_CSV.resolve()}")
    print("\n✅ Done.")


if __name__ == "__main__":
    download_kinetics_val_video_samples()
