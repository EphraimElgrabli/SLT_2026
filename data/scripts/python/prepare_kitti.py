#!/usr/bin/env python3

"""Extract the KITTI Eigen-val data and build a unified samples.jsonl.

Inputs (produced by acquire_kitti.py):
  - raw/raw_data_drives/<drive>_sync.zip   (per drive; we pull only image_02)
  - raw/data_depth_annotated.zip           (we pull only the val/ subtree)

For each line of the official val.txt (652 entries), this resolves:
  - the RGB image:  <date>/<drive>_sync/image_02/data/<frame>.png
  - the GT depth:   val/<drive>_sync/proj_depth/groundtruth/image_02/<frame>.png

and writes one record per (image, depth) pair into samples.jsonl, after
verifying both files were extracted.

KITTI GT depth is stored as 16-bit PNG; metric depth = png_value / 256.0,
with 0 meaning "no LiDAR return" (invalid). We record the scale so the
evaluator can decode consistently; we do not rewrite the PNGs.

Selective extraction keeps disk usage down: from each ~300 MB drive zip we
take only image_02/data/, and from the 13.3 GB annotated zip we take only
the val/ subtree.

Resumable: extraction is skipped when the expected files already exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

from common import resolve_root_dir


VAL_TXT_RELPATH = Path("data") / "external" / "Depth-Anything-V2" / "metric_depth" \
    / "dataset" / "splits" / "kitti" / "val.txt"

KITTI_DEPTH_SCALE = 256.0  # metric_depth_meters = png_uint16 / 256.0

DRIVE_RE = re.compile(r"raw_data/\d{4}_\d{2}_\d{2}/(\S+?_sync)/")


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "kitti"
    return {
        "dataset_root": dataset_root,
        "raw_drives_dir": dataset_root / "raw" / "raw_data_drives",
        "annotated_zip": dataset_root / "raw" / "data_depth_annotated.zip",
        "interim_dir": dataset_root / "interim",
        "raw_data_dir": dataset_root / "interim" / "raw_data",          # extracted RGB
        "annotated_dir": dataset_root / "interim" / "data_depth_annotated",  # extracted GT
        "processed_dir": dataset_root / "processed",
        "samples_manifest": dataset_root / "processed" / "samples.jsonl",
        "stats_path": dataset_root / "processed" / "stats.json",
        "val_txt": root_dir / VAL_TXT_RELPATH,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract KITTI Eigen-val data and build samples.jsonl.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract everything and rebuild the manifest from scratch.",
    )
    return parser.parse_args()


def parse_val_lines(val_txt: Path) -> list[tuple[str, str]]:
    """Return [(rgb_relpath, gt_relpath), ...] from val.txt.

    Paths in val.txt are absolute on the authors' machine
    (/mnt/bn/liheyang/Kitti/...). We strip everything up to and including
    'raw_data/' or 'data_depth_annotated/' to get a local-relative path.
    """
    pairs: list[tuple[str, str]] = []
    for line in val_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rgb_abs, gt_abs = parts[0], parts[1]
        rgb_rel = rgb_abs.split("raw_data/", 1)[1]                    # <date>/<drive>/image_02/data/<frame>.png
        gt_rel = gt_abs.split("data_depth_annotated/", 1)[1]          # val/<drive>/proj_depth/.../<frame>.png
        pairs.append((rgb_rel, gt_rel))
    return pairs


def extract_drive_images(drive_zip: Path, raw_data_dir: Path) -> None:
    """Extract only image_02/data/*.png members from one drive zip.

    Drive zips lay out as <date>/<drive>_sync/image_02/data/*.png, so the
    extracted tree under raw_data_dir mirrors the val.txt rgb relpath.
    """
    with ZipFile(drive_zip) as archive:
        members = [m for m in archive.namelist() if "/image_02/data/" in m and m.endswith(".png")]
        for member in members:
            target = raw_data_dir / member
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())


def extract_annotated_members(annotated_zip: Path, annotated_dir: Path, needed_relpaths: set[str]) -> None:
    """Extract exactly the GT PNG members referenced by val.txt.

    val.txt references frames in BOTH the train/ and val/ subtrees of
    data_depth_annotated -- KITTI's depth-prediction train/val split is NOT
    the same as the Eigen split we evaluate on -- so restricting to val/
    alone drops every frame whose GT happens to live under train/. We
    extract the specific files instead: correct and fast.
    """
    needed = {p.replace("\\", "/") for p in needed_relpaths}
    with ZipFile(annotated_zip) as archive:
        for member in archive.namelist():
            if member not in needed:
                continue
            target = annotated_dir / member
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    if not paths["val_txt"].exists():
        raise FileNotFoundError(f"Missing val.txt: {paths['val_txt']}")
    if not paths["annotated_zip"].exists():
        raise FileNotFoundError(
            f"Missing {paths['annotated_zip']}. Run `slt_data.py acquire-kitti` first."
        )

    if args.force:
        import shutil
        shutil.rmtree(paths["interim_dir"], ignore_errors=True)
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    pairs = parse_val_lines(paths["val_txt"])
    drives = sorted({DRIVE_RE.search("raw_data/" + p[0]).group(1) for p in pairs})
    print(f"  val.txt: {len(pairs)} entries across {len(drives)} drives.")

    # 1. Extract image_02 from each drive zip (idempotent, selective).
    paths["raw_data_dir"].mkdir(parents=True, exist_ok=True)
    for index, drive in enumerate(drives, start=1):
        drive_zip = paths["raw_drives_dir"] / f"{drive}.zip"
        if not drive_zip.exists():
            print(f"  WARNING: missing drive zip {drive_zip.name}; its frames will be skipped.")
            continue
        print(f"  [{index}/{len(drives)}] extracting image_02 from {drive}.zip ...")
        extract_drive_images(drive_zip, paths["raw_data_dir"])

    # 2. Extract exactly the GT depth files referenced by val.txt (idempotent).
    #    These span both the train/ and val/ subtrees of the annotated zip.
    needed_gt = {gt_rel for (_, gt_rel) in pairs}
    print(f"  Extracting {len(needed_gt)} GT depth files from data_depth_annotated.zip "
          f"(this can take a few minutes)...")
    paths["annotated_dir"].mkdir(parents=True, exist_ok=True)
    extract_annotated_members(paths["annotated_zip"], paths["annotated_dir"], needed_gt)

    # 3. Map each val.txt pair to local files and build the manifest.
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    missing_rgb: list[str] = []
    missing_gt: list[str] = []

    with paths["samples_manifest"].open("w", encoding="utf-8") as manifest:
        for rgb_rel, gt_rel in pairs:
            rgb_path = paths["raw_data_dir"] / rgb_rel
            gt_path = paths["annotated_dir"] / gt_rel
            if not rgb_path.exists():
                missing_rgb.append(rgb_rel)
                continue
            if not gt_path.exists():
                missing_gt.append(gt_rel)
                continue

            drive = DRIVE_RE.search("raw_data/" + rgb_rel).group(1)
            frame = Path(rgb_rel).stem
            record = {
                "sample_id": f"{drive}__{frame}",
                "drive": drive,
                "image_path": str(rgb_path),
                "depth_path": str(gt_path),
                "image_relpath": rgb_rel.replace("\\", "/"),
                "depth_relpath": gt_rel.replace("\\", "/"),
                "depth_scale": KITTI_DEPTH_SCALE,
                "depth_png_bits": 16,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_records.append(record)

    stats = {
        "samples": len(sample_records),
        "expected": len(pairs),
        "drives": len(drives),
        "missing_rgb_count": len(missing_rgb),
        "missing_gt_count": len(missing_gt),
        "missing_rgb_examples": missing_rgb[:5],
        "missing_gt_examples": missing_gt[:5],
        "depth_scale": KITTI_DEPTH_SCALE,
        "samples_manifest": str(paths["samples_manifest"]),
    }
    paths["stats_path"].write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()