#!/usr/bin/env python3
"""Evaluate Depth Anything V2 on the DA-2K benchmark.

For each (model, image) pair, runs inference and predicts which of the two
annotated pixels is closer to the camera based on the model's
affine-invariant inverse-depth output. Higher predicted value = closer.

Outputs:
  data/outputs/da2k/predictions/{vits,vitb,vitl}.jsonl  -- per-pair details
  data/outputs/da2k/summary.json                        -- aggregate accuracy

Resumable: if a predictions file already contains some pair_ids, those
pairs are skipped on rerun. Pass --force to start fresh.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import resolve_data_dir


# ---------------------------------------------------------------------------
# Paths and DepthAnythingV2 import (the model code lives in the cloned repo)
# ---------------------------------------------------------------------------
DATA_DIR = resolve_data_dir()
REPO_DIR = DATA_DIR / "external" / "Depth-Anything-V2"
MODELS_DIR = DATA_DIR / "models"
DA2K_DIR = DATA_DIR / "datasets" / "da2k"
OUTPUTS_DIR = DATA_DIR / "outputs" / "da2k"

if not REPO_DIR.exists():
    raise FileNotFoundError(
        f"Depth-Anything-V2 repo not found at {REPO_DIR}. "
        "Run `python data/scripts/slt_data.py bootstrap` first."
    )

sys.path.insert(0, str(REPO_DIR))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry (matches the official run.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict] = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
CHECKPOINT_FILE: dict[str, str] = {
    "vits": "depth_anything_v2_vits.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vitl": "depth_anything_v2_vitl.pth",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="Encoders to evaluate (default: all three: vits, vitb, vitl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images. Useful for sanity checks before a full run.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Inference resolution (default 518, matches the paper).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing predictions for selected models and re-run from scratch.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread variant that survives non-ASCII paths on Windows."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return image


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_image_to_pairs(da2k_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    images_path = da2k_dir / "processed" / "images.jsonl"
    pairs_path = da2k_dir / "processed" / "pairs.jsonl"
    if not images_path.exists() or not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing manifests in {da2k_dir / 'processed'}. "
            "Run `python data/scripts/slt_data.py da2k` first."
        )

    images = list(read_jsonl(images_path))
    images.sort(key=lambda r: r["image_relpath"])

    image_to_pairs: dict[str, list[dict]] = defaultdict(list)
    for pair in read_jsonl(pairs_path):
        image_to_pairs[pair["image_relpath"]].append(pair)
    return images, image_to_pairs


def load_completed_pair_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()

    completed: set[str] = set()
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed line (likely partial write from an interrupted run)
                continue
            pair_id = record.get("pair_id")
            if pair_id:
                completed.add(pair_id)
    return completed


def build_model(encoder: str, device: str) -> DepthAnythingV2:
    config = MODEL_CONFIGS[encoder]
    checkpoint_path = MODELS_DIR / CHECKPOINT_FILE[encoder]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}. "
            "Run `python data/scripts/slt_data.py bootstrap --force-checkpoint` to fetch it."
        )

    model = DepthAnythingV2(**config)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Per-model evaluation loop
# ---------------------------------------------------------------------------
def evaluate_model(
    encoder: str,
    images: list[dict],
    image_to_pairs: dict[str, list[dict]],
    predictions_path: Path,
    *,
    input_size: int,
    force: bool,
    device: str,
) -> tuple[int, float]:
    """Run inference for one model. Returns (num images processed, elapsed seconds)."""

    if force and predictions_path.exists():
        predictions_path.unlink()

    completed = load_completed_pair_ids(predictions_path)
    if completed:
        print(f"[{encoder}] Resuming: {len(completed)} pairs already in predictions file.")

    print(f"[{encoder}] Loading model checkpoint...")
    model = build_model(encoder, device)
    print(f"[{encoder}] Model ready. Running inference at {input_size}x{input_size}.")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    images_processed = 0
    start = time.time()

    with predictions_path.open("a", encoding="utf-8") as out_file:
        progress = tqdm(images, desc=encoder, unit="img")
        for image_record in progress:
            relpath = image_record["image_relpath"]
            pairs_for_image = image_to_pairs.get(relpath, [])
            todo = [p for p in pairs_for_image if p["pair_id"] not in completed]
            if not todo:
                continue

            try:
                raw = imread_unicode(Path(image_record["image_path"]))
            except FileNotFoundError as exc:
                print(f"[{encoder}] WARNING: {exc}")
                continue

            with torch.inference_mode():
                depth = model.infer_image(raw, input_size=input_size)

            depth_h, depth_w = depth.shape
            expected_h, expected_w = image_record["height"], image_record["width"]
            if depth_h != expected_h or depth_w != expected_w:
                print(
                    f"[{encoder}] WARNING: depth shape {depth.shape} != "
                    f"image shape ({expected_h}, {expected_w}) for {relpath}; skipping."
                )
                continue

            for pair in todo:
                row1, col1 = pair["closer_point_rc"]
                row2, col2 = pair["farther_point_rc"]
                pred_at_closer = float(depth[row1, col1])
                pred_at_farther = float(depth[row2, col2])
                # DAv2 outputs affine-invariant inverse depth: higher value = closer.
                correct = pred_at_closer > pred_at_farther
                record = {
                    "pair_id": pair["pair_id"],
                    "image_relpath": relpath,
                    "scene": pair["scene"],
                    "pred_at_closer": pred_at_closer,
                    "pred_at_farther": pred_at_farther,
                    "correct": bool(correct),
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed.add(pair["pair_id"])
            out_file.flush()
            images_processed += 1

    elapsed = time.time() - start
    del model
    gc.collect()
    return images_processed, elapsed


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
def aggregate_summary(predictions_dir: Path, models: list[str]) -> dict:
    summary: dict = {"models": {}, "scenes": []}
    seen_scenes: set[str] = set()

    for encoder in models:
        path = predictions_dir / f"{encoder}.jsonl"
        if not path.exists():
            continue

        per_scene_total: dict[str, int] = defaultdict(int)
        per_scene_correct: dict[str, int] = defaultdict(int)
        n_total = 0
        n_correct = 0

        for record in read_jsonl(path):
            scene = record["scene"]
            seen_scenes.add(scene)
            per_scene_total[scene] += 1
            n_total += 1
            if record["correct"]:
                per_scene_correct[scene] += 1
                n_correct += 1

        summary["models"][encoder] = {
            "n_pairs": n_total,
            "n_correct": n_correct,
            "overall_accuracy": (n_correct / n_total) if n_total else 0.0,
            "by_scene": {
                scene: {
                    "n": per_scene_total[scene],
                    "n_correct": per_scene_correct[scene],
                    "accuracy": (
                        per_scene_correct[scene] / per_scene_total[scene]
                        if per_scene_total[scene]
                        else 0.0
                    ),
                }
                for scene in sorted(per_scene_total)
            },
        }

    summary["scenes"] = sorted(seen_scenes)
    return summary


def print_console_summary(summary: dict) -> None:
    print()
    print("=" * 78)
    print("DA-2K Accuracy Summary (higher = better)")
    print("=" * 78)

    if not summary["models"]:
        print("No predictions found.")
        return

    model_names = list(summary["models"])
    header = "scene".ljust(26) + "  ".join(name.rjust(10) for name in model_names)
    print(header)
    print("-" * len(header))

    for scene in summary["scenes"]:
        row = scene.ljust(26)
        cells = []
        for name in model_names:
            entry = summary["models"][name]["by_scene"].get(scene)
            cells.append(f"{entry['accuracy'] * 100:>9.2f}%" if entry else "        -")
        row += "  ".join(cell.rjust(10) for cell in cells)
        print(row)

    print("-" * len(header))
    overall = "overall".ljust(26)
    overall_cells = [f"{summary['models'][n]['overall_accuracy'] * 100:>9.2f}%" for n in model_names]
    overall += "  ".join(cell.rjust(10) for cell in overall_cells)
    print(overall)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print(f"CPU threads available to torch: {torch.get_num_threads()}")

    images, image_to_pairs = load_image_to_pairs(DA2K_DIR)
    if args.limit is not None:
        images = images[: args.limit]
        print(f"Running with --limit: only the first {len(images)} images will be evaluated.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUTPUTS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    timing: dict[str, dict[str, float]] = {}
    for encoder in args.models:
        predictions_path = predictions_dir / f"{encoder}.jsonl"
        print(f"\n>>> {encoder}  ->  {predictions_path}")
        n_images, elapsed = evaluate_model(
            encoder=encoder,
            images=images,
            image_to_pairs=image_to_pairs,
            predictions_path=predictions_path,
            input_size=args.input_size,
            force=args.force,
            device=device,
        )
        per_image = elapsed / n_images if n_images else 0.0
        timing[encoder] = {
            "elapsed_seconds": round(elapsed, 1),
            "images_processed": n_images,
            "seconds_per_image": round(per_image, 3),
        }
        print(f"[{encoder}] {n_images} new images in {elapsed:.1f}s ({per_image:.2f}s/image)")

    summary = aggregate_summary(predictions_dir, args.models)
    summary["timing"] = timing
    summary_path = OUTPUTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_console_summary(summary)
    print(f"Full summary: {summary_path}")


if __name__ == "__main__":
    main()