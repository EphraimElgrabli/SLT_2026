#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
VENV_DIR="${DATA_DIR}/.venv"
REPO_DIR="${DATA_DIR}/external/Depth-Anything-V2"
OUTPUT_DIR="${DATA_DIR}/outputs/sanity_check"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found. Run ./data/scripts/bootstrap_depth_anything_v2.sh first." >&2
  exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Depth-Anything-V2 repository not found. Run ./data/scripts/bootstrap_depth_anything_v2.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
python run.py \
  --encoder vits \
  --img-path assets/examples \
  --outdir "${OUTPUT_DIR}" \
  --pred-only

printf '\nSanity check complete.\n'
printf 'Outputs: %s\n' "${OUTPUT_DIR}"
