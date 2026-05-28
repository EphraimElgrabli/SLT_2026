#!/usr/bin/env python3

"""Decode the NYU Depth V2 validation parquet shards into RGB + depth files.

The sayakpaul/nyu_depth_v2 validation split (654 images -- the standard
NYU-D Eigen test set) stores each row as:
  - image:     encoded RGB bytes      (640x480)
  - depth_map: encoded float32 bytes  (640x480), already in METERS

We decode each row and save:
  - processed/images/<id>.png   (RGB)
  - processed/depth/<id>.npy    (float32 depth, meters)
then write samples.jsonl pairing them.

Resumable: rows whose outputs already exist are skipped.
Requires pyarrow + Pillow in the venv.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import resolve_root_dir


SHARDS = ("validation-0000.parquet", "validation-0001.parquet")


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "nyu_depth_v2"
    return {
        "dataset_root": dataset_root,
        "raw_dir": dataset_root / "raw",
        "processed_dir": dataset_root / "processed",
        "images_dir": dataset_root / "processed" / "images",
        "depth_dir": dataset_root / "processed" / "depth",
        "samples_manifest": dataset_root / "processed" / "samples.jsonl",
        "stats_path": dataset_root / "processed" / "stats.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode NYU Depth V2 validation parquet into RGB+depth files."
    )
    parser.add_argument("--force", action="store_true", help="Re-decode even if outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    for shard in SHARDS:
        if not (paths["raw_dir"] / shard).exists():
            raise FileNotFoundError(
                f"Missing {shard} in {paths['raw_dir']}. Run `slt_data.py acquire-nyu` first."
            )

    if args.force:
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    paths["images_dir"].mkdir(parents=True, exist_ok=True)
    paths["depth_dir"].mkdir(parents=True, exist_ok=True)

    sample_records: list[dict] = []
    depth_mins: list[float] = []
    depth_maxs: list[float] = []
    index = 0

    for shard in SHARDS:
        parquet_file = pq.ParquetFile(paths["raw_dir"] / shard)
        for batch in parquet_file.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                sample_id = f"nyu_{index:04d}"
                image_out = paths["images_dir"] / f"{sample_id}.png"
                depth_out = paths["depth_dir"] / f"{sample_id}.npy"

                if not (image_out.exists() and depth_out.exists()):
                    rgb = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    depth = np.array(
                        Image.open(io.BytesIO(row["depth_map"]["bytes"])), dtype=np.float32
                    )
                    rgb.save(image_out)
                    np.save(depth_out, depth)
                else:
                    depth = np.load(depth_out)

                valid = depth > 0
                if valid.any():
                    depth_mins.append(float(depth[valid].min()))
                    depth_maxs.append(float(depth[valid].max()))

                height, width = depth.shape[:2]
                sample_records.append({
                    "sample_id": sample_id,
                    "source_shard": shard,
                    "source_image_path": row["image"].get("path"),
                    "image_path": str(image_out),
                    "depth_path": str(depth_out),
                    "image_relpath": str(image_out.relative_to(paths["dataset_root"])).replace("\\", "/"),
                    "depth_relpath": str(depth_out.relative_to(paths["dataset_root"])).replace("\\", "/"),
                    "width": width,
                    "height": height,
                })
                index += 1

    with paths["samples_manifest"].open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "samples": len(sample_records),
        "depth_min_meters": round(min(depth_mins), 3) if depth_mins else 0.0,
        "depth_max_meters": round(max(depth_maxs), 3) if depth_maxs else 0.0,
        "samples_manifest": str(paths["samples_manifest"]),
    }
    paths["stats_path"].write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()