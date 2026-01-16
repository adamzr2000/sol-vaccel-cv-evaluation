#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    me = Path(__file__).name

    scripts = sorted(
        p for p in here.glob("*.py")
        if p.name != me and p.is_file()
    )

    if not scripts:
        print("No .py scripts found to run.")
        return 0

    print(f"Running {len(scripts)} scripts in: {here}")
    for script in scripts:
        print(f"\n=== Running: {script.name} ===")
        # Run with the same interpreter (python3/venv) that launched this script
        result = subprocess.run(
            [sys.executable, script.name],
            cwd=str(here),
        )
        if result.returncode != 0:
            print(f"\nERROR: {script.name} failed with exit code {result.returncode}")
            return result.returncode

    print("\nAll plot scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

