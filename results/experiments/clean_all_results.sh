#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-tag <tag>

Removes all results for the given run-tag across:
  - docker-stats (files)
  - system-stats (files)
  - model-stats  (directories; may require sudo)
  - logs         (files/dirs matching run-tag, if present)

Example:
  ./clean_all_results.sh --run-tag run2
EOF
}

RUN_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag)
      [[ $# -ge 2 ]] || { echo "[err] --run-tag requires a value"; usage; exit 2; }
      RUN_TAG="$2"
      shift 2
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

rm_files_in_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "[clean] skip (missing): ${dir}"
    return 0
  fi

  echo "[clean] files: ${dir}"
  find "${dir}" -maxdepth 1 -type f \
    \( -name "*_${RUN_TAG}_*" -o -name "${RUN_TAG}_*" -o -name "*${RUN_TAG}*" \) \
    -print -delete || true
}

rm_model_dirs() {
  local dir="model-stats"
  if [[ ! -d "${dir}" ]]; then
    echo "[clean] skip (missing): ${dir}"
    return 0
  fi

  echo "[clean] dirs: ${dir}"

  # Collect matching dirs first (so we can retry with sudo if needed)
  mapfile -t TARGETS < <(
    find "${dir}" -maxdepth 1 -mindepth 1 -type d \
      \( -name "${RUN_TAG}_*" -o -name "*_${RUN_TAG}_*" -o -name "*${RUN_TAG}*" \) \
      -print
  )

  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "[clean] model-stats: nothing to remove"
    return 0
  fi

  printf "%s\n" "${TARGETS[@]}"

  # First try without sudo
  if rm -rf "${TARGETS[@]}" 2>/dev/null; then
    echo "[clean] model-stats removed (no sudo)"
    return 0
  fi

  # If anything still exists, retry with sudo
  local remaining=()
  for d in "${TARGETS[@]}"; do
    [[ -e "$d" ]] && remaining+=("$d")
  done

  if [[ ${#remaining[@]} -gt 0 ]]; then
    echo "[clean] model-stats: permission denied -> retry with sudo"
    sudo rm -rf "${remaining[@]}"
    echo "[clean] model-stats removed (sudo)"
  fi
}

rm_logs() {
  local dir="logs"
  if [[ ! -d "${dir}" ]]; then
    echo "[clean] skip (missing): ${dir}"
    return 0
  fi

  echo "[clean] logs: ${dir}"
  find "${dir}" -maxdepth 1 -mindepth 1 -name "*${RUN_TAG}*" -print -exec rm -rf {} + || true
}

rm_files_in_dir "docker-stats"
rm_files_in_dir "system-stats"
rm_model_dirs
rm_logs

echo "[clean] done ✅"

