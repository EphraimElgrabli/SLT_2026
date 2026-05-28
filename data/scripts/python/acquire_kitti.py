#!/usr/bin/env python3

"""Download the KITTI data needed for the Eigen 652-image val split.

Parses the official val.txt to find which raw_data drives are referenced,
then downloads only those drives (28 of them) plus the projected GT depth:

  - raw_data/<drive>_sync.zip   (per drive; carries image_02 left color cam)
  - data_depth_annotated.zip    (~13.3 GB; GT depth for train+val, already
                                 projected to the image_02 plane)

Because the GT in data_depth_annotated/val/ is pre-projected to image_02,
no camera calibration is required.

This coexists with any pre-existing data_depth_selection.zip (a different,
wrong split): it writes to a separate set of files and never deletes it.

download_file() resumes partial downloads and is safe to re-run.
Requires only stdlib + common.py (runs under system Python or the venv).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import download_file, resolve_root_dir


KITTI_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
DATA_DEPTH_ANNOTATED_URL = f"{KITTI_S3}/data_depth_annotated.zip"
DATA_DEPTH_ANNOTATED_SIZE = 14241086697

VAL_TXT_RELPATH = Path("data") / "external" / "Depth-Anything-V2" / "metric_depth" \
    / "dataset" / "splits" / "kitti" / "val.txt"

# Captures e.g. "2011_09_26_drive_0002_sync" from a raw_data/<date>/<drive>/ path.
DRIVE_RE = re.compile(r"raw_data/\d{4}_\d{2}_\d{2}/(\S+?_sync)/")


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "kitti"
    return {
        "dataset_root": dataset_root,
        "raw_dir": dataset_root / "raw",
        "raw_drives_dir": dataset_root / "raw" / "raw_data_drives",
        "val_txt": root_dir / VAL_TXT_RELPATH,
    }


def parse_unique_drives(val_txt: Path) -> list[str]:
    drives: set[str] = set()
    for line in val_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = DRIVE_RE.search(line)
        if match:
            drives.add(match.group(1))
    return sorted(drives)


def drive_zip_url(drive_sync: str) -> str:
    """Map a val.txt drive name to its KITTI S3 sync-zip URL.

    drive_sync = "2011_09_26_drive_0002_sync"
    folder      = "2011_09_26_drive_0002"   (strip trailing _sync)
    url         = .../raw_data/<folder>/<folder>_sync.zip
    """
    base = drive_sync[: -len("_sync")] if drive_sync.endswith("_sync") else drive_sync
    return f"{KITTI_S3}/raw_data/{base}/{base}_sync.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KITTI Eigen-val raw drives + annotated GT depth.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    parser.add_argument(
        "--skip-annotated",
        action="store_true",
        help="Skip the ~13.3 GB data_depth_annotated.zip (e.g. already downloaded).",
    )
    parser.add_argument(
        "--skip-drives",
        action="store_true",
        help="Skip the per-drive raw_data downloads (e.g. only need the GT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    if not paths["val_txt"].exists():
        raise FileNotFoundError(f"Missing val.txt: {paths['val_txt']}")

    drives = parse_unique_drives(paths["val_txt"])
    print(f"  Found {len(drives)} unique drives in val.txt.")

    results: list[dict] = []

    if not args.skip_drives:
        paths["raw_drives_dir"].mkdir(parents=True, exist_ok=True)
        for index, drive in enumerate(drives, start=1):
            url = drive_zip_url(drive)
            target = paths["raw_drives_dir"] / f"{drive}.zip"
            print(f"  [{index}/{len(drives)}] {drive}")
            results.append(download_file(url, target, force=args.force))

    if not args.skip_annotated:
        paths["raw_dir"].mkdir(parents=True, exist_ok=True)
        print("  Downloading data_depth_annotated.zip (~13.3 GB)...")
        results.append(download_file(
            DATA_DEPTH_ANNOTATED_URL,
            paths["raw_dir"] / "data_depth_annotated.zip",
            expected_size_bytes=DATA_DEPTH_ANNOTATED_SIZE,
            force=args.force,
        ))

    total_bytes = sum(int(r.get("size_bytes", 0)) for r in results)
    print(json.dumps(
        {"files": len(results), "total_gb": round(total_bytes / 1e9, 2)},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()