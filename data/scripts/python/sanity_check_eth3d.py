#!/usr/bin/env python3

"""Sanity-check ETH3D dense GT depth maps produced by prepare_eth3d.py.

For each chosen sample this:
  - prints coverage (% valid pixels) and the depth range in meters,
  - writes a side-by-side PNG (RGB | colorized depth) under
    processed/sanity/ so the depth structure can be eyeballed against
    the photo.

A correct projection should show: depth structure that matches the photo
(near surfaces one color, far surfaces another), a sensible metric range
(roughly 1-60 m for these outdoor/indoor scenes), and reasonable coverage
(tens of percent -- scans are dense but don't cover sky/occluded areas).

Depth colormap: a jet ramp over the 5th-95th percentile of valid depth.
Near = warm (red), far = cool (blue). Invalid pixels (no scan point) are
rendered black so coverage gaps are obvious.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import resolve_root_dir


VIEW_HEIGHT = 420  # pixels, for the side-by-side preview


def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "eth3d"
    return {
        "samples_manifest": dataset_root / "processed" / "samples.jsonl",
        "sanity_dir": dataset_root / "processed" / "sanity",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check ETH3D projected depth maps.")
    parser.add_argument("--limit", type=int, default=4, help="Number of samples to render.")
    parser.add_argument("--sample-id", default=None, help="Render one specific sample_id.")
    return parser.parse_args()


def jet_colormap(norm: np.ndarray) -> np.ndarray:
    """Map norm in [0, 1] to an (H, W, 3) uint8 jet-style RGB image."""
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    viz = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        d = depth[valid]
        lo, hi = np.percentile(d, [5, 95])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        # Invert so near = warm (red), far = cool (blue).
        viz = jet_colormap(1.0 - norm)
    viz[~valid] = 0  # black for invalid / no-scan pixels
    return viz


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    w, h = image.size
    new_w = max(1, round(w * height / h))
    return image.resize((new_w, height), Image.BILINEAR)


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    if not paths["samples_manifest"].exists():
        raise FileNotFoundError(f"Missing manifest: {paths['samples_manifest']}. Run prepare-eth3d first.")

    samples = []
    with paths["samples_manifest"].open(encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            raise ValueError(f"No sample with sample_id == {args.sample_id}")
    else:
        # Spread the picks across the manifest rather than all from one scene.
        step = max(1, len(samples) // args.limit)
        samples = samples[::step][: args.limit]

    paths["sanity_dir"].mkdir(parents=True, exist_ok=True)

    for sample in samples:
        depth = np.load(sample["depth_path"])
        valid = depth > 0
        coverage = float(valid.mean()) * 100.0
        if valid.any():
            d = depth[valid]
            dmin, dmed, dmax = float(d.min()), float(np.median(d)), float(d.max())
        else:
            dmin = dmed = dmax = 0.0

        print(
            f"  {sample['sample_id']:38}  "
            f"coverage {coverage:5.1f}%   "
            f"depth[min/med/max] = {dmin:6.2f} / {dmed:6.2f} / {dmax:7.2f} m"
        )

        rgb = Image.open(sample["image_path"]).convert("RGB")
        depth_rgb = Image.fromarray(colorize_depth(depth))

        rgb_small = resize_to_height(rgb, VIEW_HEIGHT)
        depth_small = resize_to_height(depth_rgb, VIEW_HEIGHT)

        combined = Image.new("RGB", (rgb_small.width + depth_small.width, VIEW_HEIGHT), "white")
        combined.paste(rgb_small, (0, 0))
        combined.paste(depth_small, (rgb_small.width, 0))

        out_path = paths["sanity_dir"] / f"{sample['sample_id']}.png"
        combined.save(out_path)
        print(f"      wrote {out_path}")

    print(f"\n  Wrote {len(samples)} preview(s) to {paths['sanity_dir']}")


if __name__ == "__main__":
    main()