#!/usr/bin/env python3
"""
debug_csv_durations.py  (NO ARGS)

Scans all *.csv under the current folder (recursively), excluding any "_summary/" folders.

For each CSV:
- Finds start/end timestamps (prefers timestamp_iso if present, otherwise uses timestamp ms)
- Computes duration
- Flags files with duration > 10 minutes
- Flags files spanning multiple days (start_date != end_date)
- Also reports "largest gap" between consecutive samples (helps detect appended/multi-session logs)

Run:
  python3 debug_csv_durations.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List


THRESHOLD_MINUTES = 10.0
GAP_WARN_SECONDS = 10.0  # "large gap" heuristic to spot multiple sessions appended


def _parse_iso(s: str) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        # supports "2026-01-19T18:37:37.177607+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _parse_ms(s: str) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    try:
        # timestamps may be int or float-looking
        return int(float(ss))
    except Exception:
        return None


def _dt_to_utc(dt: datetime) -> datetime:
    # If naive, assume UTC; if aware, convert to UTC.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class CsvStats:
    path: Path
    rows: int
    start_dt_utc: Optional[datetime]
    end_dt_utc: Optional[datetime]
    duration_sec: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]
    spans_multiple_days: bool
    largest_gap_sec: Optional[float]
    gap_count_over_threshold: int
    uses: str  # "timestamp_iso" or "timestamp(ms)" or "none"


def analyze_csv(path: Path) -> CsvStats:
    rows = 0
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    # For gap detection
    prev_dt: Optional[datetime] = None
    largest_gap: Optional[float] = None
    gap_count_over = 0

    uses = "none"

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        has_iso = "timestamp_iso" in fields
        has_ts = "timestamp" in fields

        # Decide primary time source
        # Prefer timestamp_iso because it’s explicit and human-meaningful.
        time_mode = "iso" if has_iso else ("ms" if has_ts else "none")

        if time_mode == "iso":
            uses = "timestamp_iso"
        elif time_mode == "ms":
            uses = "timestamp(ms)"
        else:
            uses = "none"

        for row in reader:
            rows += 1

            dt: Optional[datetime] = None
            if time_mode == "iso":
                dt = _parse_iso(row.get("timestamp_iso"))
            elif time_mode == "ms":
                ms = _parse_ms(row.get("timestamp"))
                if ms is not None:
                    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

            if dt is None:
                continue

            dt = _dt_to_utc(dt)

            if start_dt is None or dt < start_dt:
                start_dt = dt
            if end_dt is None or dt > end_dt:
                end_dt = dt

            if prev_dt is not None:
                gap = (dt - prev_dt).total_seconds()
                if gap < 0:
                    # out-of-order timestamps: treat as a big warning-style gap
                    gap = abs(gap)
                if largest_gap is None or gap > largest_gap:
                    largest_gap = gap
                if gap > GAP_WARN_SECONDS:
                    gap_count_over += 1

            prev_dt = dt

    duration_sec: Optional[float] = None
    if start_dt is not None and end_dt is not None:
        duration_sec = (end_dt - start_dt).total_seconds()

    start_date = start_dt.date().isoformat() if start_dt else None
    end_date = end_dt.date().isoformat() if end_dt else None
    spans = bool(start_date and end_date and start_date != end_date)

    return CsvStats(
        path=path,
        rows=rows,
        start_dt_utc=start_dt,
        end_dt_utc=end_dt,
        duration_sec=duration_sec,
        start_date=start_date,
        end_date=end_date,
        spans_multiple_days=spans,
        largest_gap_sec=largest_gap,
        gap_count_over_threshold=gap_count_over,
        uses=uses,
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
    all_csv = [p for p in root.rglob("*.csv") if "_summary" not in p.parts]

    if not all_csv:
        print("No CSV files found (excluding _summary/).")
        return 0

    stats: List[CsvStats] = []
    for p in sorted(all_csv):
        try:
            stats.append(analyze_csv(p))
        except Exception as e:
            print(f"ERROR reading {p}: {e}", file=sys.stderr)

    # Filter: duration > 10 min OR spans multiple days
    threshold_sec = THRESHOLD_MINUTES * 60.0
    flagged = []
    for s in stats:
        too_long = (s.duration_sec is not None and s.duration_sec > threshold_sec)
        if too_long or s.spans_multiple_days:
            flagged.append(s)

    # Sort flagged by duration desc (None last)
    def _key(s: CsvStats):
        return (-(s.duration_sec or -1), str(s.path))

    flagged.sort(key=_key)

    print(f"Scanned: {len(stats)} CSV files (excluding _summary/)")
    print(f"Flag criteria: duration > {THRESHOLD_MINUTES:.0f} min OR spans multiple days")
    print(f"Flagged: {len(flagged)}")
    print("-" * 120)

    # Header
    print(
        "duration_min | multi_day | largest_gap_s | gaps>10s | rows | time_source | start_utc                  | end_utc                    | file"
    )
    print("-" * 120)

    for s in flagged:
        dur_min = (s.duration_sec / 60.0) if s.duration_sec is not None else None
        print(
            f"{fmt_num(dur_min,2):>11} | "
            f"{'YES' if s.spans_multiple_days else 'no ':>9} | "
            f"{fmt_num(s.largest_gap_sec,3):>12} | "
            f"{str(s.gap_count_over_threshold):>7} | "
            f"{str(s.rows):>4} | "
            f"{s.uses:<11} | "
            f"{fmt_dt(s.start_dt_utc):<26} | "
            f"{fmt_dt(s.end_dt_utc):<26} | "
            f"{s.path.relative_to(root)}"
        )

    print("-" * 120)

    # Also print a quick count of multi-day files and long files
    long_count = sum(
        1 for s in stats if s.duration_sec is not None and s.duration_sec > threshold_sec
    )
    multi_day_count = sum(1 for s in stats if s.spans_multiple_days)

    print(f"Summary:")
    print(f"  > {THRESHOLD_MINUTES:.0f} min: {long_count}")
    print(f"  spans multiple days: {multi_day_count}")
    print(f"  (Tip) If you see multi-day + large gaps>10s, it's almost certainly an appended multi-session log.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

