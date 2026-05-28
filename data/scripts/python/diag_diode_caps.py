#!/usr/bin/env python3

"""Controlled experiment: how does the max_depth cap affect DIODE outdoor?

Reuses the exact alignment + metric code from evaluate_pixelwise.py, but runs
a small outdoor-only sample through several candidate caps so we can see the
DIRECTION and magnitude of the cap's effect in ~2 minutes instead of a full
24-minute run per value.

For each cap we report mean AbsRel and mean delta1 over the same sample, so the
comparison is apples-to-apples (same images, same predictions; only the cap and
the resulting valid mask / clipping change).

Run:
  python data\\scripts\\python\\diag_diode_caps.py --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Reuse the real pipeline pieces so the experiment matches production exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pixelwise import (  # noqa: E402
    BENCHMARK_CONFIG, INPUT_SIZE, MODEL_CONFIGS,
    apply_eval_crop, compute_scale_and_shift, load_model, build_paths,
)
import cv2  # noqa: E402
import torch  # noqa: E402

from common import resolve_root_dir  # noqa: E402


CANDIDATE_CAPS = [80.0, 150.0, 200.0, 300.0, 1e9]  # 1e9 ~= "no cap"


def load_diode_depth(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """DIODE depth (H,W,1)->(H,W) plus its validity mask (no cap applied yet)."""
    depth = np.squeeze(np.load(sample["depth_path"]).astype(np.float32))
    base_valid = np.isfinite(depth) & (depth > 0)
    if sample.get("mask_path"):
        m = np.squeeze(np.load(sample["mask_path"]).astype(bool))
        base_valid &= m
    return depth, base_valid


def score_with_cap(pred_disp: np.ndarray, depth: np.ndarray, base_valid: np.ndarray,
                   lo: float, hi: float) -> tuple[float, float] | None:
    """Run the exact alignment + depth-space metrics for one cap value."""
    valid = base_valid & (depth > lo) & (depth < hi)
    valid = apply_eval_crop("diode", valid)
    if valid.sum() < 100:
        return None

    target_disp = np.zeros_like(depth, dtype=np.float32)
    target_disp[valid] = 1.0 / depth[valid]
    s, t = compute_scale_and_shift(pred_disp, target_disp, valid.astype(np.float32))
    if s == 0.0 and t == 0.0:
        return None

    pred_disp_aligned = np.maximum(s * pred_disp + t, 1.0 / hi)
    pred_depth = np.clip(1.0 / pred_disp_aligned, lo, hi)

    gt = depth[valid]
    pr = pred_depth[valid]
    abs_rel = float(np.mean(np.abs(pr - gt) / gt))
    thresh = np.maximum(pr / gt, gt / pr)
    delta1 = float(np.mean((thresh < 1.25).astype(np.float32)))
    return abs_rel, delta1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="Number of outdoor samples to test.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)
    samples_file = paths["datasets_dir"] / "diode" / "processed" / "samples.jsonl"
    samples = [json.loads(l) for l in samples_file.open(encoding="utf-8") if l.strip()]
    outdoor = [s for s in samples if "outdoor" in s["sample_id"]]
    random.seed(0)
    sample = random.sample(outdoor, min(args.n, len(outdoor)))
    print(f"  DIODE outdoor cap sweep on {len(sample)} samples (ViT-S).")

    model = load_model(paths["repo_dir"], paths["models_dir"], "vits")
    lo = BENCHMARK_CONFIG["diode"]["min_depth"]

    # Cache predictions once per image; only the cap changes between conditions.
    results: dict[float, list[tuple[float, float]]] = {c: [] for c in CANDIDATE_CAPS}
    for i, s in enumerate(sample, start=1):
        depth, base_valid = load_diode_depth(s)
        bgr = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        pred = model.infer_image(bgr, input_size=INPUT_SIZE).astype(np.float32)
        if pred.shape != depth.shape:
            pred = cv2.resize(pred, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
        for cap in CANDIDATE_CAPS:
            r = score_with_cap(pred, depth, base_valid, lo, cap)
            if r is not None:
                results[cap].append(r)
        if i % 10 == 0 or i == len(sample):
            print(f"    {i}/{len(sample)}")

    print("\n  cap (m) |  mean AbsRel |  mean delta1 |  n")
    print("  --------+--------------+--------------+-----")
    for cap in CANDIDATE_CAPS:
        rs = results[cap]
        if not rs:
            print(f"  {cap:>7.0f} |     (no valid samples)")
            continue
        mar = sum(r[0] for r in rs) / len(rs)
        md1 = sum(r[1] for r in rs) / len(rs)
        label = "no cap" if cap >= 1e8 else f"{cap:.0f}"
        print(f"  {label:>7} |   {mar:8.4f}   |   {md1:8.4f}   | {len(rs)}")
    print("\n  (DIODE outdoor target overall AbsRel ~0.073; indoor already matches.)")


if __name__ == "__main__":
    main()