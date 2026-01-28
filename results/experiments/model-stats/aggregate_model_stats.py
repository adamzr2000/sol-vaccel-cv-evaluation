#!/usr/bin/env python3
"""
aggregate_model_stats.py

Aggregate model-stats benchmark_summary.json files for a given RUN_TAG.

Expected directory naming:

  Default (most backends):
    {RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{LOCAL_MODE}/benchmark_summary.json

  vAccel remote (robot -> edge) disambiguation:
    {RUN_TAG}_{MODEL}_{BACKEND}_{HOST}_{LOCAL_MODE}_target-{TARGET_DEVICE}/benchmark_summary.json

Notes:
- LOCAL_MODE reflects the local execution mode on the machine running the benchmark script
  (e.g., gpu vs cpu). For vaccel-remote on the robot, LOCAL_MODE is always "cpu".
- TARGET_DEVICE reflects the intended remote execution target for vaccel-remote sessions
  (e.g., target-cpu vs target-gpu). This suffix is used to prevent run directory collisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def discover_run_tags(run_dirs: List[Path]) -> List[str]:
    tags = set()
    for d in run_dirs:
        if not d.is_dir():
            continue
        # Directory format (minimum): RUN_TAG_MODEL_BACKEND_HOST_LOCAL_MODE[...]
        parts = d.name.split("_")
        if len(parts) >= 5:
            tags.add(parts[0])
    return sorted(tags)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-tag")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    cwd = Path(".").resolve()
    run_dirs = sorted([p for p in cwd.iterdir() if p.is_dir() and not p.name.startswith("_")])

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs in current directory:\n")
        tags = discover_run_tags(run_dirs)
        if not tags:
            print("  (none found)")
        else:
            for t in tags:
                print(f"  - {t}")
        print("\n👉 Usage:")
        print("   python3 aggregate_model_stats.py --run-tag <RUN_TAG>")
        return

    run_tag = args.run_tag
    # matched_dirs = [d for d in run_dirs if d.name.startswith(f"{run_tag}_")]
    matched_dirs = [d for d in run_dirs if d.name.split("_", 1)[0] == run_tag]
    
    if not matched_dirs:
        print(f"❌ No run directories matched RUN_TAG='{run_tag}'")
        return

    aggregated: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for d in matched_dirs:
        summary_path = d / "benchmark_summary.json"
        if not summary_path.exists():
            skipped.append(f"{d.name} (missing benchmark_summary.json)")
            continue

        try:
            with summary_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            # Keep structure unchanged; just collect objects in a list
            aggregated.append(obj)
        except Exception as e:
            skipped.append(f"{d.name} ({e})")

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_tag}_benchmark_summary.json"

    # Helpful wrapper metadata (doesn't modify individual summaries)
    output_obj = {
        "run_tag": run_tag,
        "num_runs": len(aggregated),
        "runs": aggregated,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"✅ RUN_TAG '{run_tag}' aggregated")
    print(f"📄 Output written to: {out_path}")
    print(f"📦 Runs included: {len(aggregated)}")
    if skipped:
        print(f"⚠️ Skipped: {len(skipped)}")
        for s in skipped[:10]:
            print(f"   - {s}")
        if len(skipped) > 10:
            print(f"   ... +{len(skipped) - 10} more")


if __name__ == "__main__":
    main()
