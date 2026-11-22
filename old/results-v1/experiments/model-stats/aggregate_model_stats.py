#!/usr/bin/env python3
"""
aggregate_model_stats.py

Aggregate benchmark_summary.json files by run-tag into a single JSON.

Usage:
    ./aggregate_model_stats.py --run-tag iso
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def discover_run_tags(cwd: Path) -> List[str]:
    tags: set[str] = set()
    for p in cwd.rglob("benchmark_summary.json"):
        if "failed" not in p.parts:
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

    if args.help:
        print("Usage: ./aggregate_model_stats.py --run-tag <tag>")
        return

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs:\n")
        for t in discover_run_tags(cwd) or ["  (none found)"]:
            print(f"  - {t}")
        return

    run_tag = args.run_tag.strip()

    matched_paths = sorted(
        p for p in cwd.rglob("benchmark_summary.json")
        if "failed" not in p.parts and p.parent.name.startswith(f"{run_tag}_")
    )

    if not matched_paths:
        print(f"❌ No runs found for run-tag='{run_tag}'")
        return

    print(f"🔍 Found {len(matched_paths)} runs. Aggregating...")

    aggregated: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for summary_path in matched_paths:
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if "host" not in obj:
                obj["host"] = summary_path.parent.parent.name
            aggregated.append(obj)
        except Exception as e:
            skipped.append(f"{summary_path} ({e})")

    out_dir = cwd / "_summary"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_tag}_benchmark_summary.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"run_tag": run_tag, "num_runs": len(aggregated), "runs": aggregated}, f, indent=2)

    print(f"✅ Aggregated {len(aggregated)} runs")
    print(f"📄 Output: {out_path}")

    if skipped:
        print(f"⚠️  Skipped {len(skipped)} files:")
        for s in skipped:
            print(f"   - {s}")


if __name__ == "__main__":
    main()
