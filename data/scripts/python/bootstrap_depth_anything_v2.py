#!/usr/bin/env python3

from __future__ import annotations

import argparse

from common import (
    download_file,
    python_command,
    resolve_data_dir,
    resolve_venv_python,
    run_command,
    sync_file_link,
)


PINNED_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
SMALL_CKPT_NAME = "depth_anything_v2_vits.pth"
SMALL_CKPT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
    f"{SMALL_CKPT_NAME}?download=true"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone the pinned Depth Anything V2 repository, create the venv, and download the smoke-test checkpoint."
    )
    parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Re-download the smoke-test checkpoint even if it already exists locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir()
    venv_dir = data_dir / ".venv"
    external_dir = data_dir / "external"
    repo_dir = external_dir / "Depth-Anything-V2"
    models_dir = data_dir / "models"
    checkpoints_dir = repo_dir / "checkpoints"
    checkpoint_path = models_dir / SMALL_CKPT_NAME

    for directory in (external_dir, models_dir, data_dir / "datasets", data_dir / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)

    if not (repo_dir / ".git").exists():
        run_command(["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2.git", str(repo_dir)])

    run_command(["git", "-C", str(repo_dir), "fetch", "--tags", "origin"])
    run_command(["git", "-C", str(repo_dir), "checkout", PINNED_COMMIT])

    if not venv_dir.exists():
        run_command([python_command(), "-m", "venv", str(venv_dir)])

    venv_python = resolve_venv_python(venv_dir)
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(venv_python), "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")])

    if args.force_checkpoint or not checkpoint_path.exists():
        download_file(SMALL_CKPT_URL, checkpoint_path, force=args.force_checkpoint)
    sync_file_link(checkpoint_path, checkpoints_dir / SMALL_CKPT_NAME)

    print()
    print("Bootstrap complete.")
    print(f"Repo: {repo_dir}")
    print(f"Venv: {venv_dir}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
