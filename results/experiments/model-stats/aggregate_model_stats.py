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
    for p in cwd.rglob("benchmark_summary.json"):
        parts = p.parent.name.split("_")
        if len(parts) >= 2:
            tags.add(parts[0])
    return sorted(tags)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-tag")
    # NEW: select which extra vaccel-remote family to include
    ap.add_argument("--link", choices=["5g", "wifi"], default="5g")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    cwd = Path(".").resolve()

    if args.help:
        print("Usage:")
        print("  ./aggregate_model_stats.py --run-tag <tag> [--link 5g|wifi]")
        return

    if not args.run_tag:
        print("❌ Missing required argument: --run-tag\n")
        print("📂 Available RUN_TAGs (scanned recursively):\n")
        tags = discover_run_tags(cwd)
        if not tags:
            print("  (none found)")
        else:
            for t in tags:
                print(f"  - {t}")
        return

    run_tag = args.run_tag.strip()
    link = args.link.strip()

    all_summaries = sorted(
        p for p in cwd.rglob("benchmark_summary.json")
        if "failed" not in p.parts
    )
    
    # 1) ORIGINAL behavior (unchanged):
    #    include anything whose run folder starts with "{run_tag}_"
    base_prefix = f"{run_tag}_"
    base_matched = [p for p in all_summaries if p.parent.name.startswith(base_prefix)]

    # 2) EXTENSION (new):
    #    additionally include ONLY vaccel-remote runs whose folder starts with:
    #    "{run_tag}-5g_" (default) OR "{run_tag}-wifi_"
    extra_prefix = f"{run_tag}-{link}_"
    extra_matched = [
        p for p in all_summaries
        if p.parent.name.startswith(extra_prefix) and "vaccel-remote" in p.parent.name
    ]

    # Merge + deduplicate while preserving order
    seen = set()
    matched_paths = []
    for p in base_matched + extra_matched:
        if p not in seen:
            matched_paths.append(p)
            seen.add(p)

    if not matched_paths:
        print(f"❌ No benchmark_summary.json files found for RUN_TAG='{run_tag}'")
        return

    aggregated: List[Dict[str, Any]] = []
    skipped: List[str] = []

    print(
        f"🔍 Found {len(matched_paths)} runs. "
        f"(base={len(base_matched)}, extra_vaccel_remote_{link}={len(extra_matched)}) "
        f"Aggregating..."
    )

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

    # NEW: output naming includes _5g or _wifi
    out_path = out_dir / f"{run_tag}_benchmark_summary_{link}.json"

    output_obj = {
        "run_tag": run_tag,
        "link": link,  # helpful metadata
        "num_runs": len(aggregated),
        "runs": aggregated,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"✅ RUN_TAG '{run_tag}' aggregated (+ vaccel-remote '{run_tag}-{link}_*')")
    print(f"📄 Output written to: {out_path}")

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} files due to errors.")


if __name__ == "__main__":
    main()