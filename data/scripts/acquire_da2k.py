#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import dataset_info, hf_hub_download


REPO_ID = "depth-anything/DA-2K"
ARCHIVE_NAME = "DA-2K.zip"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_root: Path
    raw_dir: Path
    archive_path: Path
    metadata_path: Path


def resolve_paths() -> DatasetPaths:
    root_dir = Path(__file__).resolve().parents[2]
    dataset_root = root_dir / "data" / "datasets" / "da2k"
    raw_dir = dataset_root / "raw"
    return DatasetPaths(
        dataset_root=dataset_root,
        raw_dir=raw_dir,
        archive_path=raw_dir / ARCHIVE_NAME,
        metadata_path=dataset_root / "metadata.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the DA-2K benchmark archive.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download from Hugging Face and overwrite the local archive.",
    )
    return parser.parse_args()


def write_metadata(paths: DatasetPaths, source_path: Path) -> None:
    info = dataset_info(REPO_ID)
    metadata = {
        "dataset": "DA-2K",
        "source": {
            "repo_id": REPO_ID,
            "repo_type": "dataset",
            "filename": ARCHIVE_NAME,
            "commit_sha": getattr(info, "sha", None),
        },
        "local": {
            "archive_path": str(paths.archive_path),
            "archive_size_bytes": paths.archive_path.stat().st_size,
            "source_cache_path": str(source_path),
        },
    }

    paths.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths()
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ARCHIVE_NAME,
            force_download=args.force,
        )
    )

    if args.force or not paths.archive_path.exists():
        shutil.copy2(source_path, paths.archive_path)

    write_metadata(paths, source_path)

    result = {
        "archive_path": str(paths.archive_path),
        "archive_size_bytes": paths.archive_path.stat().st_size,
        "metadata_path": str(paths.metadata_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
