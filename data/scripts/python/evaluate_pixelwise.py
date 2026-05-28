#!/usr/bin/env python3

"""Affine-invariant (MiDaS-style) zero-shot relative-depth evaluation.

This reproduces the protocol behind Table 2 of Depth Anything V2. The DA-V2
repo does not ship this evaluation code; per the Depth Anything V1 paper the
protocol is MiDaS's: predictions live in disparity (inverse-depth) space, are
aligned to the ground-truth disparity by a per-image least-squares scale+shift,
converted back to depth, and scored with AbsRel and delta1.

Per (benchmark, model, image):
  1. pred_disp = model.infer_image(bgr, 518)        # disparity, higher = closer
  2. resize pred_disp to GT resolution if needed
  3. build the valid mask + GT depth (per-benchmark rules)
  4. target_disp = 1 / gt_depth on valid pixels
  5. (s, t) = closed-form least squares so that s*pred_disp + t ~= target_disp
  6. pred_disp_aligned = clamp(s*pred_disp + t, min = 1/max_depth)
  7. pred_depth = clamp(1 / pred_disp_aligned, [min_depth, max_depth])
  8. AbsRel = mean(|pred-gt|/gt);  delta1 = mean(max(pred/gt, gt/pred) < 1.25)

Resumable: per-image results are appended to
outputs/pixelwise/<benchmark>_<model>.jsonl; on restart, already-scored
sample_ids are skipped. A summary.json is rewritten from the JSONL each run.

CPU-only friendly: torch thread count is capped; models load once per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from common import resolve_root_dir


# --- Model wiring (proven in evaluate_da2k.py) ----------------------------
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Per-benchmark valid depth range (meters). Used to build the valid mask and
# to clamp aligned predictions. These mirror common MiDaS-style zero-shot
# evaluation ranges; if reproduced numbers drift from the paper, revisit these.
BENCHMARK_CONFIG = {
    "nyu_depth_v2": {"min_depth": 1e-3, "max_depth": 10.0, "metric_space": "depth"},
    "kitti":        {"min_depth": 1e-3, "max_depth": 80.0, "metric_space": "depth"},
    "sintel":       {"min_depth": 1e-3, "max_depth": 400.0, "metric_space": "depth"},
    "eth3d":        {"min_depth": 1e-3, "max_depth": 80.0, "metric_space": "depth"},
    "diode":        {"min_depth": 1e-3, "max_depth": 80.0, "metric_space": "depth"},
}

ALL_BENCHMARKS = list(BENCHMARK_CONFIG.keys())
ALL_MODELS = ["vits", "vitb", "vitl"]
INPUT_SIZE = 518


# --- GT loaders -----------------------------------------------------------
def read_dpt(path: Path) -> np.ndarray:
    """Read a Sintel .dpt depth file -> (H, W) float32 (meters)."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(magic - 202021.25) > 1e-2:
            raise ValueError(f"Not a valid .dpt file: {path}")
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        data = np.fromfile(f, dtype=np.float32, count=width * height)
    return data.reshape(height, width)


def apply_eval_crop(benchmark: str, valid: np.ndarray) -> np.ndarray:
    """Intersect the valid mask with the benchmark's standard evaluation crop.

    These crops are part of the established protocol (not tuning):
      - NYU: the Eigen crop [45:471, 41:601] on 480x640 images, which removes
        the noisy Kinect border that otherwise inflates the error.
      - KITTI: the Garg crop, a fractional central region of each image.
    Other benchmarks use their full valid region.
    """
    h, w = valid.shape
    if benchmark == "nyu_depth_v2" and (h, w) == (480, 640):
        crop = np.zeros_like(valid)
        crop[45:471, 41:601] = True
        return valid & crop
    if benchmark == "kitti":
        crop = np.zeros_like(valid)
        y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
        x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
        crop[y1:y2, x1:x2] = True
        return valid & crop
    return valid


def load_gt(benchmark: str, sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt_depth_meters, valid_mask) at the GT's native resolution."""
    cfg = BENCHMARK_CONFIG[benchmark]
    lo, hi = cfg["min_depth"], cfg["max_depth"]

    if benchmark == "kitti":
        png = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED)
        depth = png.astype(np.float32) / float(sample.get("depth_scale", 256.0))
    elif benchmark in ("nyu_depth_v2", "eth3d", "diode"):
        depth = np.load(sample["depth_path"]).astype(np.float32)
    elif benchmark == "sintel":
        depth = read_dpt(Path(sample["depth_path"])).astype(np.float32)
    else:
        raise ValueError(f"Unknown benchmark {benchmark}")

    depth = np.squeeze(depth)  # DIODE stores depth as (H, W, 1); normalize to (H, W)

    valid = np.isfinite(depth) & (depth > lo) & (depth < hi)

    # DIODE ships an explicit validity mask -- intersect with it.
    if benchmark == "diode" and sample.get("mask_path"):
        m = np.squeeze(np.load(sample["mask_path"])).astype(bool)
        valid &= m

    valid = apply_eval_crop(benchmark, valid)
    return depth, valid


# --- Core MiDaS-style alignment + metrics ---------------------------------
def compute_scale_and_shift(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Closed-form least-squares (s, t) minimizing sum(mask * (s*pred + t - target)^2)."""
    a_00 = float(np.sum(mask * pred * pred))
    a_01 = float(np.sum(mask * pred))
    a_11 = float(np.sum(mask))
    b_0 = float(np.sum(mask * pred * target))
    b_1 = float(np.sum(mask * target))
    det = a_00 * a_11 - a_01 * a_01
    if det <= 0:
        return 0.0, 0.0
    s = (a_11 * b_0 - a_01 * b_1) / det
    t = (-a_01 * b_0 + a_00 * b_1) / det
    return s, t


def evaluate_sample(pred_disp: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray,
                    benchmark: str) -> dict | None:
    """Align prediction to GT disparity and compute AbsRel + delta1."""
    cfg = BENCHMARK_CONFIG[benchmark]
    lo, hi = cfg["min_depth"], cfg["max_depth"]

    if valid.sum() < 100:
        return None  # too few valid pixels to align reliably

    target_disp = np.zeros_like(gt_depth, dtype=np.float32)
    target_disp[valid] = 1.0 / gt_depth[valid]

    s, t = compute_scale_and_shift(pred_disp, target_disp, valid.astype(np.float32))
    if s == 0.0 and t == 0.0:
        return None

    pred_disp_aligned = s * pred_disp + t
    disparity_cap = 1.0 / hi
    pred_disp_aligned = np.maximum(pred_disp_aligned, disparity_cap)

    space = cfg.get("metric_space", "depth")
    if space == "disparity":
        # MiDaS evaluates ETH3D and Sintel in disparity (inverse-depth) space:
        # the relative error is computed directly on disparities, which weights
        # far (low-disparity) regions much more heavily than depth-space does.
        pr = pred_disp_aligned[valid]
        gt = target_disp[valid]
    else:
        # Depth space (metric datasets: KITTI, NYU, DIODE).
        pred_depth = 1.0 / pred_disp_aligned
        pred_depth = np.clip(pred_depth, lo, hi)
        gt = gt_depth[valid]
        pr = pred_depth[valid]

    abs_rel = float(np.mean(np.abs(pr - gt) / gt))
    thresh = np.maximum(pr / gt, gt / pr)
    delta1 = float(np.mean((thresh < 1.25).astype(np.float32)))
    return {"abs_rel": abs_rel, "delta1": delta1, "valid_px": int(valid.sum()),
            "scale": s, "shift": t}


# --- Model loading + inference --------------------------------------------
def load_model(repo_dir: Path, models_dir: Path, encoder: str):
    sys.path.insert(0, str(repo_dir))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt = models_dir / f"depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def infer_disparity(model, image_path: str, gt_shape: tuple[int, int]) -> np.ndarray | None:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    pred = model.infer_image(bgr, input_size=INPUT_SIZE)  # HxW disparity, higher=closer
    if pred.shape != gt_shape:
        pred = cv2.resize(pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_CUBIC)
    return pred.astype(np.float32)


# --- Paths + IO -----------------------------------------------------------
def build_paths(root_dir: Path) -> dict[str, Path]:
    return {
        "datasets_dir": root_dir / "data" / "datasets",
        "repo_dir": root_dir / "data" / "external" / "Depth-Anything-V2",
        "models_dir": root_dir / "data" / "models",
        "out_dir": root_dir / "data" / "outputs" / "pixelwise",
    }


def samples_path(datasets_dir: Path, benchmark: str) -> Path:
    return datasets_dir / benchmark / "processed" / "samples.jsonl"


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["sample_id"])
    return done


def write_summary(out_dir: Path) -> None:
    """Aggregate every <benchmark>_<model>.jsonl into summary.json (self-healing)."""
    summary: dict = {}
    for jsonl in sorted(out_dir.glob("*.jsonl")):
        benchmark, model = jsonl.stem.rsplit("_", 1)
        abs_rels, delta1s = [], []
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("abs_rel") is None:
                    continue
                abs_rels.append(rec["abs_rel"])
                delta1s.append(rec["delta1"])
        if abs_rels:
            summary.setdefault(benchmark, {})[model] = {
                "abs_rel": round(float(np.mean(abs_rels)), 4),
                "delta1": round(float(np.mean(delta1s)), 4),
                "n": len(abs_rels),
            }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- Main -----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Affine-invariant relative-depth evaluation (Table 2).")
    parser.add_argument("--benchmark", choices=ALL_BENCHMARKS + ["all"], default="all")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads.")
    parser.add_argument("--force", action="store_true", help="Discard existing results and re-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    models = args.models or ALL_MODELS

    for model_name in models:
        model = None  # lazy-load per model (skip if everything already done)
        for benchmark in benchmarks:
            spath = samples_path(paths["datasets_dir"], benchmark)
            if not spath.exists():
                print(f"  [skip] {benchmark}/{model_name}: no samples.jsonl")
                continue
            samples = load_samples(spath)
            if args.limit:
                samples = samples[: args.limit]

            jsonl = paths["out_dir"] / f"{benchmark}_{model_name}.jsonl"
            if args.force and jsonl.exists():
                jsonl.unlink()
            done = load_done_ids(jsonl)
            todo = [s for s in samples if s["sample_id"] not in done]
            print(f"  {benchmark}/{model_name}: {len(todo)} to do "
                  f"({len(done)} already done, {len(samples)} total)")
            if not todo:
                continue

            if model is None:
                print(f"  loading model {model_name} ...")
                model = load_model(paths["repo_dir"], paths["models_dir"], model_name)

            with jsonl.open("a", encoding="utf-8") as out:
                for i, sample in enumerate(todo, start=1):
                    gt_depth, valid = load_gt(benchmark, sample)
                    pred_disp = infer_disparity(model, sample["image_path"], gt_depth.shape)
                    if pred_disp is None:
                        rec = {"sample_id": sample["sample_id"], "abs_rel": None, "error": "image_read_failed"}
                    else:
                        metrics = evaluate_sample(pred_disp, gt_depth, valid, benchmark)
                        if metrics is None:
                            rec = {"sample_id": sample["sample_id"], "abs_rel": None, "error": "too_few_valid"}
                        else:
                            rec = {"sample_id": sample["sample_id"], **metrics}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    if i % 25 == 0 or i == len(todo):
                        print(f"    [{benchmark}/{model_name}] {i}/{len(todo)}")

            write_summary(paths["out_dir"])

    write_summary(paths["out_dir"])
    print("\n  Done. Summary:")
    print((paths["out_dir"] / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()