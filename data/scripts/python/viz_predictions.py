#!/usr/bin/env python3

"""Generate three-panel visualizations (RGB | GT | prediction) for slides.

For each requested (benchmark, sample) pair we:
  1. Load the RGB and the ground-truth depth.
  2. Run Depth Anything V2 ViT-L inference on the RGB.
  3. Affine-invariant-align the prediction to the GT (so the heat-map colors
     are comparable to the GT, otherwise the prediction's arbitrary scale
     would make them look unrelated).
  4. Render a single PNG side-by-side: RGB | GT heatmap | predicted heatmap,
     with the per-image AbsRel printed under the prediction.

The selection of which samples to render is hard-coded below as IDX_PICKS —
indices into each benchmark's samples.jsonl. The defaults are arbitrary
"middle" indices that tend to be visually interesting; feel free to edit and
re-run to pick the best frames for the deck.

Output: data/outputs/slide_figures/<benchmark>_<sample_id>.png

Run:
  python data\\scripts\\python\\viz_predictions.py
  python data\\scripts\\python\\viz_predictions.py --benchmarks nyu_depth_v2 kitti
  python data\\scripts\\python\\viz_predictions.py --model vitb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt

# Reuse the production evaluator's pieces so the visualization matches what
# the JSONLs were scored with, not some parallel reimplementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pixelwise import (  # noqa: E402
    BENCHMARK_CONFIG, INPUT_SIZE, apply_eval_crop, build_paths,
    compute_scale_and_shift, load_gt, load_model, read_dpt,
)
from common import resolve_root_dir  # noqa: E402


# Picks per benchmark. Each entry is a sample index in the benchmark's
# samples.jsonl. We use indices (not IDs) so the picks survive renames.
# Edit these if a different frame is more photogenic.
IDX_PICKS = {
    "nyu_depth_v2": [42, 100, 300],
    "kitti":        [25, 200, 400],
    "sintel":       [10, 500, 900],
    "eth3d":        [50, 200, 350],
    "diode":        None,  # populated below: 2 indoor + 2 outdoor
}


def pick_diode_indices(samples: list[dict]) -> list[int]:
    """Pick 2 indoor and 2 outdoor frames for DIODE so the deck can contrast them."""
    indoor = [i for i, s in enumerate(samples) if "indoor" in s["sample_id"]]
    outdoor = [i for i, s in enumerate(samples) if "outdoor" in s["sample_id"]]
    # Take samples from the middle of each list so they're not all the same scene.
    return [indoor[len(indoor) // 3], indoor[2 * len(indoor) // 3],
            outdoor[len(outdoor) // 3], outdoor[2 * len(outdoor) // 3]]


def colorize_depth(depth: np.ndarray, valid: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Turn a depth map into an RGB heatmap.

    Invalid pixels render dark grey so they don't blow out the color range.
    The valid pixels are normalized to [0, 1] using their own min/max so the
    heatmap is high-contrast regardless of the absolute depth range.
    """
    out = np.full((*depth.shape, 3), 0.18, dtype=np.float32)  # dark grey background
    if valid.sum() == 0:
        return (out * 255).astype(np.uint8)
    d = depth[valid]
    lo, hi = float(np.percentile(d, 2)), float(np.percentile(d, 98))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(norm)[..., :3].astype(np.float32)
    out[valid] = colored[valid]
    return (out * 255).astype(np.uint8)


def render_panel(rgb: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray,
                 pred_depth: np.ndarray, pred_valid: np.ndarray,
                 abs_rel: float | None, title: str, out_path: Path) -> None:
    """Save a 3-panel figure: RGB | GT heatmap | prediction heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].imshow(rgb[..., ::-1])  # BGR -> RGB for matplotlib
    axes[0].set_title("Input image (RGB)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(colorize_depth(gt_depth, gt_valid))
    axes[1].set_title("Ground truth depth", fontsize=11)
    axes[1].axis("off")

    pred_title = "Our prediction"
    if abs_rel is not None:
        pred_title += f"   (AbsRel = {abs_rel:.3f})"
    axes[2].imshow(colorize_depth(pred_depth, pred_valid))
    axes[2].set_title(pred_title, fontsize=11)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_sample(model, benchmark: str, sample: dict, out_dir: Path) -> None:
    """Run inference, align, render a 3-panel figure for one sample."""
    cfg = BENCHMARK_CONFIG[benchmark]
    lo, hi = cfg["min_depth"], cfg["max_depth"]

    bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  skip (image missing): {sample['sample_id']}")
        return
    depth_gt, valid_gt = load_gt(benchmark, sample)
    if valid_gt.sum() < 100:
        print(f"  skip (too few valid GT pixels): {sample['sample_id']}")
        return

    # Inference and resize to GT resolution
    pred = model.infer_image(bgr, input_size=INPUT_SIZE).astype(np.float32)
    if pred.shape != depth_gt.shape:
        pred = cv2.resize(pred, (depth_gt.shape[1], depth_gt.shape[0]),
                          interpolation=cv2.INTER_CUBIC)

    # Affine-invariant alignment, matching evaluate_pixelwise
    valid_eval = apply_eval_crop(benchmark, valid_gt)
    target_disp = np.zeros_like(depth_gt, dtype=np.float32)
    target_disp[valid_eval] = 1.0 / depth_gt[valid_eval]
    s, t = compute_scale_and_shift(pred, target_disp, valid_eval.astype(np.float32))
    pred_disp_aligned = np.maximum(s * pred + t, 1.0 / hi)
    pred_depth = np.clip(1.0 / pred_disp_aligned, lo, hi)

    abs_rel = None
    if valid_eval.sum() >= 100:
        gt = depth_gt[valid_eval]
        pr = pred_depth[valid_eval]
        abs_rel = float(np.mean(np.abs(pr - gt) / gt))

    safe_id = sample["sample_id"].replace("/", "__").replace("\\", "__")
    title = f"{benchmark}  -  {safe_id}"
    out_path = out_dir / f"{benchmark}_{safe_id}.png"
    render_panel(bgr, depth_gt, valid_eval, pred_depth, valid_eval, abs_rel, title, out_path)
    extra = f"AbsRel={abs_rel:.3f}" if abs_rel is not None else "AbsRel=N/A"
    print(f"  wrote: {out_path.name}   ({extra})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        default=["nyu_depth_v2", "kitti", "diode"],
                        help="Which benchmarks to visualize. Defaults to the three "
                             "most photogenic; add 'sintel eth3d' for the rest.")
    parser.add_argument("--model", default="vitl",
                        help="Model size to visualize (vits / vitb / vitl). Default vitl "
                             "since that's the headline result.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    root = resolve_root_dir()
    paths = build_paths(root)
    out_dir = root / "data" / "outputs" / "slide_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output dir: {out_dir}")

    model = load_model(paths["repo_dir"], paths["models_dir"], args.model)

    for benchmark in args.benchmarks:
        samples_path = paths["datasets_dir"] / benchmark / "processed" / "samples.jsonl"
        if not samples_path.exists():
            print(f"  skip {benchmark}: samples.jsonl not found")
            continue
        samples = [json.loads(l) for l in samples_path.open(encoding="utf-8") if l.strip()]
        picks = IDX_PICKS[benchmark] if benchmark != "diode" else pick_diode_indices(samples)
        print(f"\n  {benchmark}: rendering {len(picks)} samples (model {args.model})")
        for idx in picks:
            if 0 <= idx < len(samples):
                process_sample(model, benchmark, samples[idx], out_dir)
            else:
                print(f"  skip: index {idx} out of range (have {len(samples)} samples)")

    print(f"\n  Done. Open: {out_dir}")


if __name__ == "__main__":
    main()