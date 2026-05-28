#!/usr/bin/env python3

"""Extract and validate the DIODE val benchmark.

Walks the extracted directory tree and writes one record per RGB-D triplet
(image + depth + mask) into a JSONL manifest, plus an overall stats JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

from common import resolve_root_dir


ARCHIVE_NAME = "val.tar.gz"
DATASET_FOLDER_NAME = "val"
SCENE_TYPES = ("indoors", "outdoor")


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "diode"
    interim_dir = dataset_root / "interim"
    processed_dir = dataset_root / "processed"
    return {
        "dataset_root": dataset_root,
        "archive_path": dataset_root / "raw" / ARCHIVE_NAME,
        "interim_dir": interim_dir,
        "extracted_root": interim_dir / DATASET_FOLDER_NAME,
        "processed_dir": processed_dir,
        "samples_manifest": processed_dir / "samples.jsonl",
        "stats_path": processed_dir / "stats.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate the DIODE val benchmark.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete previous extracted and processed outputs before rebuilding them.",
    )
    return parser.parse_args()


def extract_archive(archive_path: Path, interim_dir: Path, extracted_root: Path, force: bool) -> None:
    if force:
        shutil.rmtree(interim_dir, ignore_errors=True)

    if extracted_root.exists():
        return

    interim_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {archive_path.name} -> {interim_dir} (this can take a couple of minutes)...")
    with tarfile.open(archive_path, "r:gz") as archive:
        # filter='data' (Python 3.12+) blocks absolute paths, '..' traversal,
        # and unsafe permissions / symlinks during extraction.
        archive.extractall(path=interim_dir, filter="data")
    print("  Extraction complete.")


def to_posix_relpath(absolute_path: Path, base_path: Path) -> str:
    return absolute_path.relative_to(base_path).as_posix()


def build_manifest(extracted_root: Path, processed_dir: Path) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_manifest_path = processed_dir / "samples.jsonl"

    scene_type_counter: Counter[str] = Counter()
    resolution_counter: Counter[tuple[int, int]] = Counter()
    missing_depth: list[str] = []
    missing_mask: list[str] = []
    total_samples = 0

    with samples_manifest_path.open("w", encoding="utf-8") as samples_file:
        for scene_type in SCENE_TYPES:
            scene_type_root = extracted_root / scene_type
            if not scene_type_root.exists():
                print(f"  Warning: expected directory not found: {scene_type_root}")
                continue

            for image_path in sorted(scene_type_root.rglob("*.png")):
                stem = image_path.stem  # filename without ".png"
                depth_path = image_path.parent / f"{stem}_depth.npy"
                mask_path = image_path.parent / f"{stem}_depth_mask.npy"

                image_relpath = to_posix_relpath(image_path, extracted_root)

                if not depth_path.exists():
                    missing_depth.append(image_relpath)
                    continue
                if not mask_path.exists():
                    missing_mask.append(image_relpath)
                    continue

                with Image.open(image_path) as image:
                    width, height = image.size

                sample_id = image_relpath[: -len(image_path.suffix)]  # strip trailing ".png"
                sample_record = {
                    "sample_id": sample_id,
                    "scene_type": scene_type,
                    "image_relpath": image_relpath,
                    "depth_relpath": to_posix_relpath(depth_path, extracted_root),
                    "mask_relpath": to_posix_relpath(mask_path, extracted_root),
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "mask_path": str(mask_path),
                    "width": width,
                    "height": height,
                }
                samples_file.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                scene_type_counter[scene_type] += 1
                resolution_counter[(width, height)] += 1
                total_samples += 1

    return {
        "samples": total_samples,
        "scene_types": dict(scene_type_counter),
        "resolutions": {f"{w}x{h}": count for (w, h), count in resolution_counter.most_common()},
        "missing_depth_count": len(missing_depth),
        "missing_mask_count": len(missing_mask),
        "missing_depth_examples": missing_depth[:5],
        "missing_mask_examples": missing_mask[:5],
        "samples_manifest": str(samples_manifest_path),
    }


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    archive_path = paths["archive_path"]
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing archive: {archive_path}. Run the acquisition step first."
        )

    if args.force:
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    extract_archive(
        archive_path=archive_path,
        interim_dir=paths["interim_dir"],
        extracted_root=paths["extracted_root"],
        force=args.force,
    )

    stats = build_manifest(paths["extracted_root"], paths["processed_dir"])
    paths["stats_path"].write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()