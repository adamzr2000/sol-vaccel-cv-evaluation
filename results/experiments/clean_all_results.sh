#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-tag <tag> [--dry-run]

Deletes ONLY result artifacts for the given run-tag:
  - docker-stats/**: matching *.csv (recursive)
  - system-stats/**: matching *.csv (recursive)
  - model-stats/**: matching run directories (recursive) + matching summary *.json/*.csv
Does NOT touch:
  - logs/
  - any .py/.sh/README/.gitkeep/.gitignore/etc.

Example:
  ./clean_all_results.sh --run-tag run1
  ./clean_all_results.sh --run-tag run1 --dry-run
EOF
}

RUN_TAG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag)
      [[ $# -ge 2 ]] || { echo "[err] --run-tag requires a value"; usage; exit 2; }
      RUN_TAG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[err] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_TAG}" ]]; then
  echo "[err] Missing --run-tag"
  usage
  exit 2
fi

echo "[clean] run-tag=${RUN_TAG}"
echo "[clean] root=$(pwd)"
[[ "$DRY_RUN" -eq 1 ]] && echo "[clean] DRY-RUN (no deletions)"

# Simplified: Look for the tag preceded by / or _ or -
# And followed by . or _ or - or end of string.
NAME_EXPR=(-regextype posix-extended -regex ".*[/_-]${RUN_TAG}([._-].*|$)")

do_rm() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cat
  else
    xargs -0r rm -f --
  fi
}

do_rmdir_rf() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cat
  else
    xargs -0r rm -rf --
  fi
}

rm_csv_recursive() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo "[clean] skip (missing): $dir"; return 0; }
  echo "[clean] csv (recursive, excluding failed/): $dir"

  # We use -print0 INSIDE the parentheses to ensure matched items are output
  find "$dir" -path "*/failed" -prune -o \( -type f -name "*.csv" "${NAME_EXPR[@]}" -print0 \) |
    tee >(tr '\0' '\n' >&2) |
    do_rm >/dev/null
}

rm_model_stats() {
  local dir="model-stats"
  [[ -d "$dir" ]] || { echo "[clean] skip (missing): $dir"; return 0; }
  echo "[clean] model-stats (excluding failed/): $dir"

  # 1. Clean Directories (excluding 'failed')
  # Note: we use -mindepth 1 so we don't accidentally match the root 'model-stats' dir itself
  find "$dir" -mindepth 1 -path "*/failed" -prune -o \( -type d "${NAME_EXPR[@]}" -print0 \) |
    tee >(tr '\0' '\n' >&2) |
    do_rmdir_rf >/dev/null

  # 2. Clean Summary files (excluding 'failed')
  find "$dir" -path "*/failed" -prune -o \( -type f \( -name "*.json" -o -name "*.csv" \) "${NAME_EXPR[@]}" -print0 \) |
    tee >(tr '\0' '\n' >&2) |
    do_rm >/dev/null
}

rm_csv_recursive "docker-stats"
rm_csv_recursive "system-stats"
rm_model_stats

echo "[clean] done ✅ (logs untouched)"
