#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_DIRS=(
  "configs/experiments"
  "data"
  "results"
)

DRY_RUN=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: scripts/flush_repo_files.sh [--dry-run] [--force]

Deletes all files inside:
  - configs/experiments
  - data
  - results

Directories are preserved.

Options:
  --dry-run   Show files that would be deleted without deleting them
  --force     Skip confirmation prompt
  -h, --help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

FILES_TO_DELETE=()

for dir in "${TARGET_DIRS[@]}"; do
  ABS_DIR="${ROOT_DIR}/${dir}"
  if [[ -d "${ABS_DIR}" ]]; then
    while IFS= read -r file; do
      FILES_TO_DELETE+=("${file}")
    done < <(find "${ABS_DIR}" -type f | sort)
  fi
done

if [[ ${#FILES_TO_DELETE[@]} -eq 0 ]]; then
  echo "No files found in target directories."
  exit 0
fi

echo "Files selected for deletion (${#FILES_TO_DELETE[@]}):"
for file in "${FILES_TO_DELETE[@]}"; do
  echo "  ${file#${ROOT_DIR}/}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo
  echo "Dry-run completed. No files were deleted."
  exit 0
fi

if [[ "${FORCE}" != "true" ]]; then
  echo
  read -r -p "Proceed with deletion? [y/N] " reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

for file in "${FILES_TO_DELETE[@]}"; do
  rm -f "${file}"
done

echo "Done. Deleted ${#FILES_TO_DELETE[@]} files. Directory structure preserved."
