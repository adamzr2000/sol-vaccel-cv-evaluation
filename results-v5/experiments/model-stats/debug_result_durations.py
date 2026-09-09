#!/usr/bin/env python3
"""
debug_benchmark_summaries.py (NO ARGS)

Scans all */benchmark_summary.json under the current folder (recursively),
excluding any "_summary/" folders.

Checks:
- duration_sec (from JSON)
- computed duration (stop-start)
- flags: duration > 10 min OR spans multiple days OR duration mismatch OR missing fields

Run:
  python3 debug_benchmark_summaries.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple


THRESHOLD_MINUTES = 10.0
DURATION_TOLERANCE_SEC = 0.250  # allow small rounding differences


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    try:
        # supports "2026-01-19T18:30:01.706434+00:00"
        dt = datetime.fromisoformat(ss)
        # normalize to UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class SummaryStats:
    path: Path
    run_id: str
    backend: str
    host: str
    model: str
    device: str
    num_samples: Optional[int]

    start_utc: Optional[datetime]
    stop_utc: Optional[datetime]
    duration_reported_s: Optional[float]
    duration_computed_s: Optional[float]

    spans_multiple_days: bool
    duration_mismatch_s: Optional[float]
    invalid: bool
    reason: str


def analyze_summary(path: Path) -> SummaryStats:
    run_id = backend = host = model = device = ""
    num_samples: Optional[int] = None

    start_utc = stop_utc = None
    duration_reported_s = duration_computed_s = None
    spans = False
    mismatch = None
    invalid = False
    reason = ""

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return SummaryStats(
            path=path,
            run_id="",
            backend="",
            host="",
            model="",
            device="",
            num_samples=None,
            start_utc=None,
            stop_utc=None,
            duration_reported_s=None,
            duration_computed_s=None,
            spans_multiple_days=False,
            duration_mismatch_s=None,
            invalid=True,
            reason=f"JSON parse error: {e}",
        )

    run_id = str(data.get("run_id", "")).strip()
    backend = str(data.get("backend", "")).strip()
    host = str(data.get("host", "")).strip()
    model = str(data.get("model", "")).strip()
    device = str(data.get("device", "")).strip()
    try:
        ns = data.get("num_samples", None)
        num_samples = int(ns) if ns is not None else None
    except Exception:
        num_samples = None

    tw = data.get("time_window", {}) or {}
    start_utc = _parse_iso(tw.get("start"))
    stop_utc = _parse_iso(tw.get("stop"))
    duration_reported_s = _safe_float(tw.get("duration_sec"))

    # compute duration from timestamps
    if start_utc and stop_utc:
        duration_computed_s = (stop_utc - start_utc).total_seconds()
        spans = (start_utc.date().isoformat() != stop_utc.date().isoformat())
    else:
        duration_computed_s = None

    # mismatch check
    if duration_reported_s is not None and duration_computed_s is not None:
        mismatch = abs(duration_reported_s - duration_computed_s)
    else:
        mismatch = None

    # validity / reason
    missing_fields = []
    if not run_id: missing_fields.append("run_id")
    if not backend: missing_fields.append("backend")
    if not host: missing_fields.append("host")
    if not model: missing_fields.append("model")
    if not device: missing_fields.append("device")
    if start_utc is None: missing_fields.append("time_window.start")
    if stop_utc is None: missing_fields.append("time_window.stop")
    if duration_reported_s is None: missing_fields.append("time_window.duration_sec")

    if missing_fields:
        invalid = True
        reason = "missing/invalid: " + ", ".join(missing_fields)
    elif mismatch is not None and mismatch > DURATION_TOLERANCE_SEC:
        # not necessarily invalid, but worth flagging
        reason = f"duration mismatch {mismatch:.3f}s"
    else:
        reason = ""

    return SummaryStats(
        path=path,
        run_id=run_id,
        backend=backend,
        host=host,
        model=model,
        device=device,
        num_samples=num_samples,
        start_utc=start_utc,
        stop_utc=stop_utc,
        duration_reported_s=duration_reported_s,
        duration_computed_s=duration_computed_s,
        spans_multiple_days=spans,
        duration_mismatch_s=mismatch,
        invalid=invalid,
        reason=reason,
    )


def fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.isoformat().replace("+00:00", "Z")


def fmt_num(x: Optional[float], decimals: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{decimals}f}"


def main() -> int:
    root = Path(".").resolve()
    summaries = [
        p for p in root.rglob("benchmark_summary.json")
        if "_summary" not in p.parts
    ]

    if not summaries:
        print("No benchmark_summary.json files found (excluding _summary/).")
        return 0

    stats: List[SummaryStats] = []
    for p in sorted(summaries):
        try:
            stats.append(analyze_summary(p))
        except Exception as e:
            print(f"ERROR analyzing {p}: {e}", file=sys.stderr)

    threshold_sec = THRESHOLD_MINUTES * 60.0

    def is_flagged(s: SummaryStats) -> bool:
        # duration basis: prefer computed, fallback to reported
        dur = s.duration_computed_s if s.duration_computed_s is not None else s.duration_reported_s
        too_long = (dur is not None and dur > threshold_sec)
        mismatch = (s.duration_mismatch_s is not None and s.duration_mismatch_s > DURATION_TOLERANCE_SEC)
        return s.invalid or s.spans_multiple_days or too_long or mismatch

    flagged = [s for s in stats if is_flagged(s)]

    # sort flagged by duration desc, invalid last-ish
    def sort_key(s: SummaryStats) -> Tuple[int, float]:
        dur = s.duration_computed_s if s.duration_computed_s is not None else (s.duration_reported_s or -1.0)
        return (0 if not s.invalid else 1, -dur)

    flagged.sort(key=sort_key)

    print(f"Scanned: {len(stats)} benchmark_summary.json files (excluding _summary/)")
    print(f"Flag criteria: duration > {THRESHOLD_MINUTES:.0f} min OR spans multiple days OR duration mismatch > {DURATION_TOLERANCE_SEC:.3f}s OR invalid")
    print(f"Flagged: {len(flagged)}")
    print("-" * 140)
    print(
        "duration_min | multi_day | mismatch_s | invalid | backend       | host  | device | model                      | start_utc                  | stop_utc                   | run_id / folder"
    )
    print("-" * 140)

    for s in flagged:
        dur_s = s.duration_computed_s if s.duration_computed_s is not None else s.duration_reported_s
        dur_min = (dur_s / 60.0) if dur_s is not None else None

        rel = s.path.parent.relative_to(root)  # folder name
        inv = "YES" if s.invalid else "no "
        md = "YES" if s.spans_multiple_days else "no "
        mm = fmt_num(s.duration_mismatch_s, 3)

        print(
            f"{fmt_num(dur_min,2):>11} | "
            f"{md:>9} | "
            f"{mm:>9} | "
            f"{inv:>7} | "
            f"{s.backend:<12} | "
            f"{s.host:<5} | "
            f"{s.device:<6} | "
            f"{(s.model[:26] + ('..' if len(s.model) > 26 else '')):<26} | "
            f"{fmt_dt(s.start_utc):<26} | "
            f"{fmt_dt(s.stop_utc):<26} | "
            f"{rel}"
            + (f"  ({s.reason})" if s.reason else "")
        )

    print("-" * 140)

    long_count = 0
    multi_day_count = 0
    mismatch_count = 0
    invalid_count = 0

    for s in stats:
        dur_s = s.duration_computed_s if s.duration_computed_s is not None else s.duration_reported_s
        if dur_s is not None and dur_s > threshold_sec:
            long_count += 1
        if s.spans_multiple_days:
            multi_day_count += 1
        if s.duration_mismatch_s is not None and s.duration_mismatch_s > DURATION_TOLERANCE_SEC:
            mismatch_count += 1
        if s.invalid:
            invalid_count += 1

    print("Summary:")
    print(f"  > {THRESHOLD_MINUTES:.0f} min: {long_count}")
    print(f"  spans multiple days: {multi_day_count}")
    print(f"  duration mismatch > {DURATION_TOLERANCE_SEC:.3f}s: {mismatch_count}")
    print(f"  invalid/missing fields: {invalid_count}")
    if invalid_count or multi_day_count or mismatch_count:
        print("  (Tip) If you see multi-day or big mismatches, it's usually an overwritten/edited file or mixed sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

