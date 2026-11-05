#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    me = Path(__file__).name

    scripts = sorted(
        p for p in here.glob("barplot_*.py")
        if p.name != me and p.is_file()
    )

    if not scripts:
        print("No barplot_*.py scripts found.")
        return 0

    print(f"Running {len(scripts)} barplot scripts in: {here}")

    for script in scripts:
        print(f"\n=== Running: {script.name} ===")

        result = subprocess.run(
            [sys.executable, script.name],
            cwd=here,
        )

        if result.returncode != 0:
            print(
                f"\nERROR: {script.name} failed "
                f"with exit code {result.returncode}"
            )
            return result.returncode

    print("\nAll barplot scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())