#!/usr/bin/env python3
"""
aggregate_model_stats.py
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

def discover_run_tags(cwd: Path) -> List[str]:
    tags = set()
    # ### CHANGED: Look recursively for json files, check parent folder for tag
    for p in cwd.rglob("benchmark_summary.json"):
        parts = p.parent.name.split("_")
        if len(parts) >= 2:
            tags.add(parts[0])
    return sorted(tags)

def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-tag")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    cwd = Path(".").resolve()

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs (scanned recursively):\n")
        tags = discover_run_tags(cwd)
        if not tags: print("  (none found)")
        else:
            for t in tags: print(f"  - {t}")
        return

    run_tag = args.run_tag.strip()
    
    # ### CHANGED: Recursively find all summary files
    all_summaries = sorted(cwd.rglob("benchmark_summary.json"))
    
    # Filter: Keep only those where the parent folder starts with run_tag
    matched_paths = [p for p in all_summaries if p.parent.name.startswith(f"{run_tag}_")]

    if not matched_paths:
        print(f"❌ No benchmark_summary.json files found for RUN_TAG='{run_tag}'")
        return

    aggregated: List[Dict[str, Any]] = []
    skipped: List[str] = []

    print(f"🔍 Found {len(matched_paths)} runs. Aggregating...")

    for summary_path in matched_paths:
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            
            # benchmark_summary.json -> run_folder -> host_folder (grandparent)
            obj["host"] = summary_path.parent.parent.name
            
            aggregated.append(obj)
        except Exception as e:
            skipped.append(f"{summary_path} ({e})")

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_tag}_benchmark_summary.json"

    output_obj = {
        "run_tag": run_tag,
        "num_runs": len(aggregated),
        "runs": aggregated,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"✅ RUN_TAG '{run_tag}' aggregated")
    print(f"📄 Output written to: {out_path}")

if __name__ == "__main__":
    main()