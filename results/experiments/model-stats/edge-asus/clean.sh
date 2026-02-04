#!/usr/bin/env bash
set -euo pipefail

echo "Removing .csv/.json files..."
find . -type f \( -name "*.csv" -o -name "*.json" \) -print -delete

echo "Removing empty directories..."
# -depth ensures we delete deepest dirs first
# skip "." itself
find . -depth -type d -empty ! -path . -print -delete

