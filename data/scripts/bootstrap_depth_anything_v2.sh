#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
VENV_DIR="${DATA_DIR}/.venv"
EXTERNAL_DIR="${DATA_DIR}/external"
REPO_DIR="${EXTERNAL_DIR}/Depth-Anything-V2"
MODELS_DIR="${DATA_DIR}/models"
CHECKPOINTS_DIR="${REPO_DIR}/checkpoints"
PINNED_COMMIT="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
SMALL_CKPT_NAME="depth_anything_v2_vits.pth"
SMALL_CKPT_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/${SMALL_CKPT_NAME}?download=true"

mkdir -p "${EXTERNAL_DIR}" "${MODELS_DIR}" "${DATA_DIR}/datasets" "${DATA_DIR}/outputs"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git "${REPO_DIR}"
fi

git -C "${REPO_DIR}" fetch --tags origin
git -C "${REPO_DIR}" checkout "${PINNED_COMMIT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_DIR}/requirements.txt"

mkdir -p "${CHECKPOINTS_DIR}"

if [[ ! -f "${MODELS_DIR}/${SMALL_CKPT_NAME}" ]]; then
  curl -L "${SMALL_CKPT_URL}" -o "${MODELS_DIR}/${SMALL_CKPT_NAME}"
fi

ln -sf "${MODELS_DIR}/${SMALL_CKPT_NAME}" "${CHECKPOINTS_DIR}/${SMALL_CKPT_NAME}"

printf '\nBootstrap complete.\n'
printf 'Repo: %s\n' "${REPO_DIR}"
printf 'Venv: %s\n' "${VENV_DIR}"
printf 'Checkpoint: %s\n' "${MODELS_DIR}/${SMALL_CKPT_NAME}"
