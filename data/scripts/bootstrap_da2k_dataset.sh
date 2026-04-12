#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/data/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found. Run ./data/scripts/bootstrap_depth_anything_v2.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

python "${ROOT_DIR}/data/scripts/acquire_da2k.py"
python "${ROOT_DIR}/data/scripts/prepare_da2k.py"
