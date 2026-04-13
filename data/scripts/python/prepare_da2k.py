#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from common import resolve_root_dir


ARCHIVE_NAME = "DA-2K.zip"
DATASET_FOLDER_NAME = "DA-2K"


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "da2k"
    interim_dir = dataset_root / "interim"
    processed_dir = dataset_root / "processed"
    return {
        "dataset_root": dataset_root,
        "archive_path": dataset_root / "raw" / ARCHIVE_NAME,
        "interim_dir": interim_dir,
        "extracted_root": interim_dir / DATASET_FOLDER_NAME,
        "processed_dir": processed_dir,
        "images_manifest": processed_dir / "images.jsonl",
        "pairs_manifest": processed_dir / "pairs.jsonl",
        "stats_path": processed_dir / "stats.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate the DA-2K benchmark.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete previous extracted and processed outputs before rebuilding them.",
    )
    return parser.parse_args()


def should_skip_member(member_name: str) -> bool:
    parts = Path(member_name).parts
    return (
        "__MACOSX" in parts
        or any(part == ".DS_Store" for part in parts)
        or any(part.startswith("._") for part in parts)
    )


def extract_archive(archive_path: Path, interim_dir: Path, extracted_root: Path, force: bool) -> None:
    if force:
        shutil.rmtree(interim_dir, ignore_errors=True)

    if extracted_root.exists():
        return

    interim_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for member_name in archive.namelist():
            if should_skip_member(member_name):
                continue

            target_path = interim_dir / member_name
            if member_name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def normalize_pair(annotation: dict, width: int, height: int) -> dict:
    point1 = annotation["point1"]
    point2 = annotation["point2"]
    closer_point = annotation["closer_point"]

    if closer_point not in {"point1", "point2"}:
        raise ValueError(f"Unsupported closer_point value: {closer_point}")

    closer = point1 if closer_point == "point1" else point2
    farther = point2 if closer_point == "point1" else point1

    return {
        "closer_point_rc": closer,
        "farther_point_rc": farther,
        "closer_point_xy_norm": [round(closer[1] / width, 6), round(closer[0] / height, 6)],
        "farther_point_xy_norm": [round(farther[1] / width, 6), round(farther[0] / height, 6)],
    }


def validate_point(point: list[int], width: int, height: int, field_name: str, image_path: str) -> None:
    if len(point) != 2:
        raise ValueError(f"{field_name} must contain two coordinates for {image_path}")

    row, col = point
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"{field_name} is out of image bounds for {image_path}: row={row}, col={col}, height={height}, width={width}"
        )


def extract_ascii_prefix(file_name: str) -> str:
    match = re.match(r"[A-Za-z0-9_.() -]+", Path(file_name).stem)
    return match.group(0) if match else ""


def resolve_image_path(extracted_root: Path, image_relpath: str) -> tuple[Path, str | None]:
    image_path = extracted_root / image_relpath
    if image_path.exists():
        return image_path, None

    parent_dir = image_path.parent
    if not parent_dir.exists():
        raise FileNotFoundError(f"Missing image directory referenced by annotations: {parent_dir}")

    requested_name = Path(image_relpath).name
    requested_prefix = extract_ascii_prefix(requested_name)
    candidates = [
        candidate
        for candidate in parent_dir.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() == image_path.suffix.lower()
        and extract_ascii_prefix(candidate.name) == requested_prefix
    ]

    if len(candidates) == 1:
        actual_path = candidates[0]
        actual_relpath = str(actual_path.relative_to(extracted_root))
        return actual_path, actual_relpath

    raise FileNotFoundError(f"Missing image referenced by annotations: {image_relpath}")


def build_manifests(extracted_root: Path, processed_dir: Path) -> dict:
    annotations_path = extracted_root / "annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_manifest_path = processed_dir / "images.jsonl"
    pairs_manifest_path = processed_dir / "pairs.jsonl"

    scene_counter: Counter[str] = Counter()
    pair_counter: Counter[int] = Counter()
    total_pairs = 0
    path_corrections: list[dict[str, str]] = []

    with images_manifest_path.open("w", encoding="utf-8") as images_file, pairs_manifest_path.open(
        "w", encoding="utf-8"
    ) as pairs_file:
        for image_relpath in sorted(annotations):
            image_path, actual_image_relpath = resolve_image_path(extracted_root, image_relpath)

            with Image.open(image_path) as image:
                width, height = image.size

            scene = Path(image_relpath).parts[1]
            pairs = annotations[image_relpath]

            scene_counter[scene] += 1
            pair_counter[len(pairs)] += 1
            total_pairs += len(pairs)

            image_record = {
                "image_relpath": image_relpath,
                "resolved_image_relpath": actual_image_relpath or image_relpath,
                "image_path": str(image_path),
                "scene": scene,
                "width": width,
                "height": height,
                "pair_count": len(pairs),
            }
            images_file.write(json.dumps(image_record, ensure_ascii=False) + "\n")

            if actual_image_relpath is not None:
                path_corrections.append(
                    {
                        "annotation_image_relpath": image_relpath,
                        "resolved_image_relpath": actual_image_relpath,
                    }
                )

            for pair_index, annotation in enumerate(pairs):
                validate_point(annotation["point1"], width, height, "point1", image_relpath)
                validate_point(annotation["point2"], width, height, "point2", image_relpath)
                pair_record = {
                    "pair_id": f"{image_relpath}#{pair_index}",
                    "image_relpath": image_relpath,
                    "scene": scene,
                    "width": width,
                    "height": height,
                    **normalize_pair(annotation, width, height),
                }
                pairs_file.write(json.dumps(pair_record, ensure_ascii=False) + "\n")

    return {
        "images": len(annotations),
        "pairs": total_pairs,
        "scenes": dict(scene_counter),
        "pairs_per_image": {str(key): value for key, value in sorted(pair_counter.items())},
        "path_corrections": path_corrections,
        "images_manifest": str(images_manifest_path),
        "pairs_manifest": str(pairs_manifest_path),
    }


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    archive_path = paths["archive_path"]
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing archive: {archive_path}. Run `python data/scripts/slt_data.py da2k` first."
        )

    if args.force:
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    extract_archive(
        archive_path=archive_path,
        interim_dir=paths["interim_dir"],
        extracted_root=paths["extracted_root"],
        force=args.force,
    )

    stats = build_manifests(paths["extracted_root"], paths["processed_dir"])
    paths["stats_path"].write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
