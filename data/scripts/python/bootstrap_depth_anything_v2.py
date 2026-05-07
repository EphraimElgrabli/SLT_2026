#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass

from common import (
    download_file,
    python_command,
    resolve_data_dir,
    resolve_venv_python,
    run_command,
    sync_file_link,
)


PINNED_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    filename: str
    repo_id: str


# Note: Depth Anything V2 publicly releases the Small, Base, and Large student
# models only. The Giant teacher model has not been released by the authors,
# so it is intentionally not listed here.
CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        key="vits",
        filename="depth_anything_v2_vits.pth",
        repo_id="depth-anything/Depth-Anything-V2-Small",
    ),
    CheckpointSpec(
        key="vitb",
        filename="depth_anything_v2_vitb.pth",
        repo_id="depth-anything/Depth-Anything-V2-Base",
    ),
    CheckpointSpec(
        key="vitl",
        filename="depth_anything_v2_vitl.pth",
        repo_id="depth-anything/Depth-Anything-V2-Large",
    ),
)


def checkpoint_url(spec: CheckpointSpec) -> str:
    return f"https://huggingface.co/{spec.repo_id}/resolve/main/{spec.filename}?download=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone the pinned Depth Anything V2 repository, create the venv, and download the V2 student checkpoints."
    )
    parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Re-download the checkpoints even if they already exist locally.",
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

    print()
    print(f"Downloading {len(CHECKPOINTS)} checkpoint(s): {', '.join(spec.key for spec in CHECKPOINTS)}")
    downloaded_paths: list[str] = []
    for spec in CHECKPOINTS:
        target_path = models_dir / spec.filename
        if args.force_checkpoint or not target_path.exists():
            download_file(checkpoint_url(spec), target_path, force=args.force_checkpoint)
        sync_file_link(target_path, checkpoints_dir / spec.filename)
        downloaded_paths.append(str(target_path))

    print()
    print("Bootstrap complete.")
    print(f"Repo: {repo_dir}")
    print(f"Venv: {venv_dir}")
    print("Checkpoints:")
    for path in downloaded_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()