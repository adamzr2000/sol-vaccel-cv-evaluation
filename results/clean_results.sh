#!/bin/bash
set -euo pipefail

# Safety: ensure we are in the results directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob

targets=( *_cpu *_gpu )

if ((${#targets[@]} == 0)); then
  echo "Nothing to delete."
  exit 0
fi

echo "Removing:"
printf '  %s\n' "${targets[@]}"

rm -rf -- "${targets[@]}"

rm combined_latency_summary.json
