#!/usr/bin/env python3
"""Evaluate Depth Anything V2 metric checkpoints on local benchmark assets."""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import struct
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from common import resolve_data_dir


DATA_DIR = resolve_data_dir()
METRIC_REPO_DIR = DATA_DIR / "external" / "Depth-Anything-V2" / "metric_depth"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs" / "metric_depth"

if not METRIC_REPO_DIR.exists():
    raise FileNotFoundError("Metric-depth repo code is missing. Run `python data/scripts/slt_data.py bootstrap` first.")

sys.path.insert(0, str(METRIC_REPO_DIR))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

CHECKPOINTS = {
    "indoor_vits": {
        "encoder": "vits",
        "filename": "depth_anything_v2_metric_hypersim_vits.pth",
        "max_depth": 20.0,
    },
    "outdoor_vits": {
        "encoder": "vits",
        "filename": "depth_anything_v2_metric_vkitti_vits.pth",
        "max_depth": 80.0,
    },
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    image_bgr: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray
    checkpoint_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["kitti", "nyu_depth_v2", "sintel", "eth3d", "diode"],
        default=["kitti", "nyu_depth_v2", "sintel", "eth3d", "diode"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset sample limit.")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--median-align", action="store_true", help="Apply per-image median scaling before metrics.")
    return parser.parse_args()


def decode_png(data: bytes, flags: int) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if decoded is None:
        raise ValueError("Could not decode PNG bytes.")
    return decoded


def load_model(checkpoint_key: str, device: str) -> DepthAnythingV2:
    spec = CHECKPOINTS[checkpoint_key]
    checkpoint_path = MODELS_DIR / spec["filename"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing metric checkpoint {checkpoint_path}. "
            "Run `python data/scripts/slt_data.py metric-checkpoints` first."
        )
    model = DepthAnythingV2(**MODEL_CONFIGS[spec["encoder"]], max_depth=spec["max_depth"])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    return model.to(device).eval()


def iter_kitti(limit: int | None) -> Iterator[Sample]:
    archive_path = DATA_DIR / "datasets" / "kitti" / "raw" / "data_depth_selection.zip"
    if not archive_path.exists():
        return
    with zipfile.ZipFile(archive_path) as archive:
        image_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("depth_selection/val_selection_cropped/image/") and name.endswith(".png")
        )
        for index, image_name in enumerate(image_names):
            if limit is not None and index >= limit:
                break
            depth_name = image_name.replace("/image/", "/groundtruth_depth/").replace("_sync_image_", "_sync_groundtruth_depth_")
            if depth_name not in archive.NameToInfo:
                continue
            image = decode_png(archive.read(image_name), cv2.IMREAD_COLOR)
            depth = decode_png(archive.read(depth_name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            valid = (depth > 0) & (depth < 80)
            yield Sample("kitti", Path(image_name).name, image, depth, valid, "outdoor_vits")


def iter_nyu(limit: int | None) -> Iterator[Sample]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("NYU Depth V2 evaluation requires h5py in the local venv.") from exc

    mat_path = DATA_DIR / "datasets" / "nyu_depth_v2" / "raw" / "nyu_depth_v2_labeled.mat"
    if not mat_path.exists():
        return
    with h5py.File(mat_path, "r") as h5:
        images = h5["images"]
        depths = h5["depths"]
        count = images.shape[0] if limit is None else min(limit, images.shape[0])
        for index in range(count):
            rgb = np.asarray(images[index]).transpose(2, 1, 0)
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = np.asarray(depths[index]).transpose(1, 0).astype(np.float32)
            valid = (depth > 0) & (depth < 10) & np.isfinite(depth)
            yield Sample("nyu_depth_v2", f"{index:04d}", image, depth, valid, "indoor_vits")


def read_sintel_dpt(data: bytes) -> np.ndarray:
    stream = io.BytesIO(data)
    tag = struct.unpack("f", stream.read(4))[0]
    if abs(tag - 202021.25) > 1e-4:
        raise ValueError("Invalid Sintel .dpt tag.")
    width = struct.unpack("i", stream.read(4))[0]
    height = struct.unpack("i", stream.read(4))[0]
    depth = np.frombuffer(stream.read(width * height * 4), dtype=np.float32)
    return depth.reshape((height, width))


def iter_sintel(limit: int | None) -> Iterator[Sample]:
    image_archive_path = DATA_DIR / "datasets" / "sintel" / "raw" / "MPI-Sintel-complete.zip"
    depth_archive_path = DATA_DIR / "datasets" / "sintel" / "raw" / "MPI-Sintel-depth-training-20150305.zip"
    if not image_archive_path.exists() or not depth_archive_path.exists():
        return
    with zipfile.ZipFile(image_archive_path) as image_archive, zipfile.ZipFile(depth_archive_path) as depth_archive:
        image_names = sorted(
            name
            for name in image_archive.namelist()
            if name.startswith("training/final/") and name.endswith(".png")
        )
        yielded = 0
        for image_name in image_names:
            depth_suffix = image_name.replace("training/final/", "training/depth/").removesuffix(".png") + ".dpt"
            depth_name = next((name for name in (depth_suffix, f"MPI-Sintel-depth-training-20150305/{depth_suffix}") if name in depth_archive.NameToInfo), None)
            if depth_name is None:
                continue
            if limit is not None and yielded >= limit:
                break
            image = decode_png(image_archive.read(image_name), cv2.IMREAD_COLOR)
            depth = read_sintel_dpt(depth_archive.read(depth_name)).astype(np.float32)
            valid = (depth > 0) & (depth < 1000) & np.isfinite(depth)
            yielded += 1
            yield Sample("sintel", image_name.removeprefix("training/final/"), image, depth, valid, "indoor_vits")


def iter_diode(limit: int | None) -> Iterator[Sample]:
    archive_path = DATA_DIR / "datasets" / "diode" / "raw" / "val.tar.gz"
    if not archive_path.exists():
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        pending: dict[str, dict[str, bytes]] = {}
        yielded = 0
        for member in archive:
            if not member.isfile():
                continue
            if limit is not None and yielded >= limit:
                break
            if not (member.name.endswith(".png") or member.name.endswith("_depth.npy") or member.name.endswith("_depth_mask.npy")):
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            if member.name.endswith("_depth_mask.npy"):
                base_name = member.name.removesuffix("_depth_mask.npy")
                key = "mask"
            elif member.name.endswith("_depth.npy"):
                base_name = member.name.removesuffix("_depth.npy")
                key = "depth"
            else:
                base_name = member.name.removesuffix(".png")
                key = "image"
            current = pending.setdefault(base_name, {})
            current[key] = file_obj.read()
            if not {"image", "depth", "mask"}.issubset(current):
                continue
            image = decode_png(current["image"], cv2.IMREAD_COLOR)
            depth = np.load(io.BytesIO(current["depth"])).astype(np.float32).squeeze()
            mask = np.load(io.BytesIO(current["mask"])).astype(bool).squeeze()
            valid = mask & (depth > 0) & (depth < 1000) & np.isfinite(depth)
            image_name = f"{base_name}.png"
            checkpoint_key = "indoor_vits" if "/indoors/" in image_name else "outdoor_vits"
            yielded += 1
            pending.pop(base_name, None)
            yield Sample("diode", image_name, image, depth, valid, checkpoint_key)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def ensure_eth3d_interim() -> Path:
    archive_path = DATA_DIR / "datasets" / "eth3d" / "raw" / "multi_view_training_dslr_jpg.7z"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing ETH3D archive: {archive_path}")

    interim_dir = DATA_DIR / "datasets" / "eth3d" / "interim" / "dslr_jpg"
    marker_path = interim_dir / ".extract-complete"
    if marker_path.exists():
        return interim_dir

    interim_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "7z",
            "x",
            str(archive_path),
            f"-o{interim_dir}",
            "-y",
            "*/dslr_calibration_jpg/*",
            "*/images/dslr_images/*",
        ],
        check=True,
    )
    marker_path.write_text("ok\n", encoding="utf-8")
    return interim_dir


def read_eth3d_points(points_path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with points_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            points[point_id] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return points


def read_eth3d_images(images_path: Path) -> Iterator[tuple[np.ndarray, np.ndarray, str, list[tuple[float, float, int]]]]:
    with images_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    index = 0
    while index + 1 < len(lines):
        metadata = lines[index].split()
        observations = lines[index + 1].split()
        qvec = np.array([float(value) for value in metadata[1:5]], dtype=np.float64)
        tvec = np.array([float(value) for value in metadata[5:8]], dtype=np.float64)
        image_name = metadata[9]
        triples: list[tuple[float, float, int]] = []
        for obs_index in range(0, len(observations), 3):
            point_id = int(observations[obs_index + 2])
            if point_id < 0:
                continue
            triples.append((float(observations[obs_index]), float(observations[obs_index + 1]), point_id))
        yield qvec, tvec, image_name, triples
        index += 2


def eth3d_checkpoint_for_scene(scene_name: str) -> str:
    return "indoor_vits" if scene_name == "office" else "outdoor_vits"


def iter_eth3d(limit: int | None) -> Iterator[Sample]:
    root = ensure_eth3d_interim()
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    yielded = 0
    for scene_dir in scene_dirs:
        calibration_dir = scene_dir / "dslr_calibration_jpg"
        image_dir = scene_dir / "images"
        images_path = calibration_dir / "images.txt"
        points_path = calibration_dir / "points3D.txt"
        if not images_path.exists() or not points_path.exists():
            continue
        points = read_eth3d_points(points_path)
        for qvec, tvec, image_name, observations in read_eth3d_images(images_path):
            if limit is not None and yielded >= limit:
                return
            image_path = image_dir / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            depth = np.zeros((height, width), dtype=np.float32)
            valid = np.zeros((height, width), dtype=bool)
            rotation = qvec_to_rotmat(qvec)
            for x, y, point_id in observations:
                point = points.get(point_id)
                if point is None:
                    continue
                col = int(round(x))
                row = int(round(y))
                if row < 0 or row >= height or col < 0 or col >= width:
                    continue
                z = float((rotation @ point + tvec)[2])
                if not np.isfinite(z) or z <= 0 or z >= 1000:
                    continue
                if not valid[row, col] or z < depth[row, col]:
                    depth[row, col] = z
                    valid[row, col] = True
            if int(valid.sum()) < 25:
                continue
            yielded += 1
            yield Sample(
                "eth3d",
                f"{scene_dir.name}/{image_name}",
                image,
                depth,
                valid,
                eth3d_checkpoint_for_scene(scene_dir.name),
            )


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    pred = np.clip(pred.astype(np.float64), eps, None)
    target = np.clip(target.astype(np.float64), eps, None)
    ratio = np.maximum(target / pred, pred / target)
    diff = pred - target
    return {
        "abs_rel": float(np.mean(np.abs(diff) / target)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def evaluate_dataset(
    name: str,
    samples: Iterator[Sample],
    *,
    input_size: int,
    median_align: bool,
    device: str,
    records_path: Path,
) -> dict:
    models: dict[str, DepthAnythingV2] = {}
    records = []
    totals = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}

    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as records_file:
        for index, sample in enumerate(samples, start=1):
            model = models.get(sample.checkpoint_key)
            if model is None:
                model = load_model(sample.checkpoint_key, device)
                models[sample.checkpoint_key] = model

            with torch.inference_mode():
                pred = model.infer_image(sample.image_bgr, input_size=input_size)

            if pred.shape != sample.depth_m.shape:
                pred = cv2.resize(pred, (sample.depth_m.shape[1], sample.depth_m.shape[0]), interpolation=cv2.INTER_CUBIC)

            valid = sample.valid_mask & np.isfinite(pred) & (pred > 0)
            pred_valid = pred[valid]
            target_valid = sample.depth_m[valid]
            if pred_valid.size == 0:
                continue
            if median_align:
                scale = np.median(target_valid) / max(np.median(pred_valid), 1e-6)
                pred_valid = pred_valid * scale

            metric_values = compute_metrics(pred_valid, target_valid)
            for key, value in metric_values.items():
                totals[key] += value
            record = {
                "sample_id": sample.sample_id,
                "checkpoint": sample.checkpoint_key,
                "valid_pixels": int(pred_valid.size),
                **metric_values,
            }
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index == 1 or index % 25 == 0:
                print(f"  {name}: processed {index} samples", flush=True)
            records_file.flush()

    for model in models.values():
        del model

    n = len(records)
    return {
        "dataset": name,
        "n_samples": n,
        "median_align": median_align,
        "metrics": {key: (value / n if n else math.nan) for key, value in totals.items()},
        "records_path": str(records_path),
    }


def sample_iterator(dataset: str, limit: int | None) -> Iterator[Sample]:
    if dataset == "kitti":
        return iter_kitti(limit)
    if dataset == "nyu_depth_v2":
        return iter_nyu(limit)
    if dataset == "sintel":
        return iter_sintel(limit)
    if dataset == "eth3d":
        return iter_eth3d(limit)
    if dataset == "diode":
        return iter_diode(limit)
    raise ValueError(dataset)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"datasets": {}}
    else:
        summary = {"datasets": {}}
    summary["device"] = device
    summary["input_size"] = args.input_size
    summary.setdefault("datasets", {})
    for dataset in args.datasets:
        print(f"Evaluating {dataset}...")
        result = evaluate_dataset(
            dataset,
            sample_iterator(dataset, args.limit),
            input_size=args.input_size,
            median_align=args.median_align,
            device=device,
            records_path=OUTPUT_DIR / f"{dataset}_records.jsonl",
        )
        detail_path = OUTPUT_DIR / f"{dataset}_details.json"
        detail_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["datasets"][dataset] = {
            "n_samples": result["n_samples"],
            "metrics": result["metrics"],
            "details_path": str(detail_path),
        }
        print(json.dumps(summary["datasets"][dataset], ensure_ascii=False, indent=2))

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Metric summary: {summary_path}")


if __name__ == "__main__":
    main()
