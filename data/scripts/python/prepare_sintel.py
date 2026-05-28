#!/usr/bin/env python3

"""Extract and validate the MPI-Sintel depth benchmark.

Extracts the flow zip (which carries the RGB images) and the depth zip
(which carries the GT depth maps), then builds a samples.jsonl manifest
pairing each training/final/ RGB frame with its training/depth/ .dpt
counterpart.

Following the MDE community convention (MiDaS, Depth Anything), we evaluate
on the 'final' pass of the training set -- the harder pass with motion blur
and atmospheric effects. The test set has no public ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from common import resolve_root_dir


COMPLETE_ARCHIVE = "MPI-Sintel-complete.zip"
DEPTH_ARCHIVE = "MPI-Sintel-depth-training-20150305.zip"
EVAL_PASS = "final"  # other valid value: "clean"


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "sintel"
    raw_dir = dataset_root / "raw"
    interim_dir = dataset_root / "interim"
    processed_dir = dataset_root / "processed"
    return {
        "dataset_root": dataset_root,
        "complete_archive_path": raw_dir / COMPLETE_ARCHIVE,
        "depth_archive_path": raw_dir / DEPTH_ARCHIVE,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "samples_manifest": processed_dir / "samples.jsonl",
        "stats_path": processed_dir / "stats.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate the MPI-Sintel depth benchmark.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete previous extracted and processed outputs before rebuilding them.",
    )
    return parser.parse_args()


def find_subdir(interim_dir: Path, relative_path: str) -> Path | None:
    """Find a directory like 'training/final' anywhere in interim_dir.

    Tries up to two levels deep, to handle archives that extract either
    directly into interim_dir/training/... or into
    interim_dir/<archive-name>/training/...
    """
    direct = interim_dir / relative_path
    if direct.is_dir():
        return direct
    if not interim_dir.exists():
        return None
    for child in interim_dir.iterdir():
        if not child.is_dir():
            continue
        candidate = child / relative_path
        if candidate.is_dir():
            return candidate
    return None


def extract_zip_if_missing(archive_path: Path, target_dir: Path, expected_subdir: str) -> None:
    """Extract archive into target_dir, unless expected_subdir is already present."""
    if find_subdir(target_dir, expected_subdir) is not None:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} -> {target_dir} (this can take several minutes)...")
    with ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    print(f"  {archive_path.name} extraction complete.")


def build_manifest(interim_dir: Path, processed_dir: Path) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_manifest_path = processed_dir / "samples.jsonl"

    final_root = find_subdir(interim_dir, f"training/{EVAL_PASS}")
    depth_root = find_subdir(interim_dir, "training/depth")
    if final_root is None:
        raise FileNotFoundError(
            f"Could not locate training/{EVAL_PASS}/ under {interim_dir}. "
            f"Did the complete archive extract correctly?"
        )
    if depth_root is None:
        raise FileNotFoundError(
            f"Could not locate training/depth/ under {interim_dir}. "
            f"Did the depth archive extract correctly?"
        )

    print(f"  RGB {EVAL_PASS} pass: {final_root.relative_to(interim_dir)}")
    print(f"  Depth GT:       {depth_root.relative_to(interim_dir)}")

    sequence_counter: Counter[str] = Counter()
    resolution_counter: Counter[tuple[int, int]] = Counter()
    missing_depth: list[str] = []
    total_samples = 0

    with samples_manifest_path.open("w", encoding="utf-8") as samples_file:
        for image_path in sorted(final_root.rglob("frame_*.png")):
            sequence = image_path.parent.name
            depth_path = depth_root / sequence / (image_path.stem + ".dpt")

            image_relpath = image_path.relative_to(interim_dir).as_posix()

            if not depth_path.exists():
                missing_depth.append(image_relpath)
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            sample_id = f"{sequence}/{image_path.stem}"
            sample_record = {
                "sample_id": sample_id,
                "sequence": sequence,
                "pass_name": EVAL_PASS,
                "image_relpath": image_relpath,
                "depth_relpath": depth_path.relative_to(interim_dir).as_posix(),
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "width": width,
                "height": height,
            }
            samples_file.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
            sequence_counter[sequence] += 1
            resolution_counter[(width, height)] += 1
            total_samples += 1

    return {
        "samples": total_samples,
        "pass_name": EVAL_PASS,
        "sequences": dict(sequence_counter),
        "resolutions": {f"{w}x{h}": count for (w, h), count in resolution_counter.most_common()},
        "missing_depth_count": len(missing_depth),
        "missing_depth_examples": missing_depth[:5],
        "samples_manifest": str(samples_manifest_path),
    }


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    for label, archive_path in (
        ("complete", paths["complete_archive_path"]),
        ("depth", paths["depth_archive_path"]),
    ):
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Missing {label} archive: {archive_path}. "
                f"Run `python data\\scripts\\python\\acquire_remaining_benchmarks.py --benchmarks sintel` first."
            )

    if args.force:
        shutil.rmtree(paths["interim_dir"], ignore_errors=True)
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    extract_zip_if_missing(
        archive_path=paths["complete_archive_path"],
        target_dir=paths["interim_dir"],
        expected_subdir=f"training/{EVAL_PASS}",
    )

    extract_zip_if_missing(
        archive_path=paths["depth_archive_path"],
        target_dir=paths["interim_dir"],
        expected_subdir="training/depth",
    )

    stats = build_manifest(paths["interim_dir"], paths["processed_dir"])
    paths["stats_path"].write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()