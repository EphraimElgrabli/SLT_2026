#!/usr/bin/env python3

"""Extract ETH3D archives and build dense GT depth maps by projecting the
laser scan point clouds onto each DSLR image.

ETH3D's high-resolution multi-view training set ships as two 7z archives:
  - multi_view_training_dslr_jpg.7z       (DSLR JPGs + COLMAP calibration)
  - multi_view_training_dslr_scan_eval.7z (laser scan PLYs + alignment)

Each scene is scanned from MULTIPLE laser positions (scan1.ply, scan2.ply,
...). scan_alignment.mlp (a MeshLab project) holds one 4x4 transform per
scan, mapping that scan's local frame into the COLMAP/world frame the
camera poses live in. We load every scan, transform each by its OWN matched
matrix, merge them, and project the merged cloud -- otherwise coverage is
partial (or zero, if only the wrong scan/matrix is used).

The shipped calibration uses the COLMAP THIN_PRISM_FISHEYE camera model
(fisheye-style equidistant radial + tangential + thin-prism distortion),
which we apply when projecting so depth maps line up with the JPGs.

Pipeline:
  1. Extract both archives under interim/ (idempotent).
  2. Discover scenes via dslr_calibration_jpg/cameras.txt.
  3. Per scene: load + merge all scans (each transformed by its matrix),
     then for each DSLR image project the merged cloud (with distortion),
     z-buffer per pixel, and save a dense depth map .npy (0 = invalid).
  4. Write samples.jsonl pairing each DSLR image with its depth file.

Resumable: per-image .npy outputs are skipped if already present.
Requires plyfile and py7zr in the venv.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET

import numpy as np
import py7zr
from plyfile import PlyData

from common import resolve_root_dir


JPG_ARCHIVE = "multi_view_training_dslr_jpg.7z"
SCAN_ARCHIVE = "multi_view_training_dslr_scan_eval.7z"


# ===========================================================================
# COLMAP parsing
# ===========================================================================
class Camera(NamedTuple):
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Distortion params, layout depends on `model`.
    # PINHOLE / SIMPLE_PINHOLE: empty tuple
    # THIN_PRISM_FISHEYE: (k1, k2, p1, p2, k3, k4, sx1, sy1)
    distortion: tuple[float, ...]


class ImagePose(NamedTuple):
    image_id: int
    qvec: tuple[float, float, float, float]   # (qw, qx, qy, qz), world->camera
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


def parse_cameras_txt(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]

        if model == "PINHOLE":
            fx, fy, cx, cy = params
            distortion: tuple[float, ...] = ()
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params
            fx = fy = f
            distortion = ()
        elif model == "THIN_PRISM_FISHEYE":
            # COLMAP canonical: f, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 (11 params).
            # Some COLMAP forks emit fx, fy separately for 12 params. Handle both.
            if len(params) == 11:
                f, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
                fx = fy = f
            elif len(params) == 12:
                fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
            else:
                raise ValueError(
                    f"Unexpected THIN_PRISM_FISHEYE parameter count {len(params)} "
                    f"in {path} (expected 11 or 12)."
                )
            distortion = (k1, k2, p1, p2, k3, k4, sx1, sy1)
        else:
            raise ValueError(f"Unsupported COLMAP camera model {model} in {path}")

        cameras[camera_id] = Camera(camera_id, model, width, height, fx, fy, cx, cy, distortion)
    return cameras


def parse_images_txt(path: Path) -> list[ImagePose]:
    poses: list[ImagePose] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Header line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = int(parts[0])
        qvec = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        tvec = (float(parts[5]), float(parts[6]), float(parts[7]))
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        poses.append(ImagePose(image_id, qvec, tvec, camera_id, name))
        i += 1   # The next line is the POINTS2D row, which we ignore.
    return poses


def quaternion_to_rotation(qvec: tuple[float, float, float, float]) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix (world->camera)."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw    ],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw    ],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


# ===========================================================================
# PLY + alignment
# ===========================================================================
def load_ply_xyz(ply_path: Path) -> np.ndarray:
    """Read XYZ coordinates from a binary or ASCII .ply. Returns (N, 3) float32."""
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    return np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
        axis=1,
    ).astype(np.float32)


def parse_scan_alignment(mlp_path: Path) -> dict[str, np.ndarray]:
    """Return {ply_filename: 4x4 transform} from a MeshLab .mlp project.

    Each MLMesh element carries a filename/label and an MLMatrix44 placing
    that scan into the project (= COLMAP/world) coordinate frame.
    """
    transforms: dict[str, np.ndarray] = {}
    if not mlp_path.exists():
        return transforms
    tree = ET.parse(mlp_path)
    for mesh in tree.getroot().iter("MLMesh"):
        filename = mesh.get("filename") or mesh.get("label")
        matrix_elem = mesh.find("MLMatrix44")
        if filename is None or matrix_elem is None or matrix_elem.text is None:
            continue
        values = matrix_elem.text.strip().split()
        if len(values) != 16:
            continue
        # Use just the basename so it matches the on-disk .ply name.
        key = Path(filename).name
        transforms[key] = np.array(values, dtype=np.float64).reshape(4, 4)
    return transforms


def load_and_merge_scans(info: dict, scene_name: str) -> np.ndarray:
    """Load every scan PLY for a scene, transform each by its own matrix
    from scan_alignment.mlp, and concatenate into a single (N, 3) cloud.
    """
    transforms = parse_scan_alignment(info.get("scan_alignment_mlp", Path("does-not-exist")))
    merged: list[np.ndarray] = []
    for scan_ply in sorted(info["scan_plys"]):
        pts = load_ply_xyz(scan_ply)
        matrix = transforms.get(scan_ply.name)
        if matrix is not None and not np.allclose(matrix, np.eye(4)):
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts, ones], axis=1)
            pts = (matrix.astype(np.float32) @ pts_h.T).T[:, :3].astype(np.float32)
            tag = "transformed"
        elif matrix is not None:
            tag = "identity matrix"
        else:
            tag = "NO MATRIX in .mlp -- using as-is"
        print(f"  [{scene_name}]   {scan_ply.name}: {pts.shape[0]:,} points ({tag})")
        merged.append(pts)
    points = np.concatenate(merged, axis=0)
    print(f"  [{scene_name}] merged {len(merged)} scan(s) -> {points.shape[0]:,} points total.")
    return points


# ===========================================================================
# THIN_PRISM_FISHEYE distortion
# ===========================================================================
def apply_thin_prism_fisheye_distortion(
    x: np.ndarray,
    y: np.ndarray,
    distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply COLMAP THIN_PRISM_FISHEYE distortion to normalized image-plane coords.

    x, y are X/Z, Y/Z (camera-frame projected onto the z=1 plane).
    Returns distorted normalized coords (x_d, y_d). Pixel position is then
    fx * x_d + cx, fy * y_d + cy.

    Model: equidistant radial (fisheye) + tangential + thin-prism.
      r       = sqrt(x^2 + y^2)
      theta   = atan(r)
      theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
      radial:     scale = theta_d / r   (limit 1 at r=0)
      tangential: 2*p1*xy + p2*(r^2 + 2x^2), p1*(r^2 + 2y^2) + 2*p2*xy
      prism:      sx1*r^2, sy1*r^2
    """
    k1, k2, p1, p2, k3, k4, sx1, sy1 = distortion
    r2 = x * x + y * y
    r = np.sqrt(r2)
    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta2 * theta4
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    # scale = theta_d / r, with limit 1 at r=0.
    safe_r = np.where(r > 1e-12, r, 1.0)
    scale = np.where(r > 1e-12, theta_d / safe_r, 1.0)
    x_radial = x * scale
    y_radial = y * scale
    # Tangential
    xy = x * y
    x_tang = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
    y_tang = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
    # Thin prism
    x_prism = sx1 * r2
    y_prism = sy1 * r2
    return x_radial + x_tang + x_prism, y_radial + y_tang + y_prism


# ===========================================================================
# Projection
# ===========================================================================
def project_scan_to_depth(
    points_world: np.ndarray,
    R_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """Project (N, 3) world-frame points through camera, z-buffer per pixel.

    Returns a float32 (H, W) depth map. Pixels with no projected point
    get depth = 0 (used as the implicit "invalid" marker).
    """
    H, W = camera.height, camera.width
    points_cam = (R_world_to_cam @ points_world.T).T + t_world_to_cam

    # Drop points behind the camera AND those so close that X/Z, Y/Z
    # would blow up and produce NaN downstream (e.g. inf * 0 = NaN).
    NEAR_CLIP = 0.01  # meters
    keep = points_cam[:, 2] > NEAR_CLIP
    if not keep.any():
        return np.zeros((H, W), dtype=np.float32)
    points_cam = points_cam[keep]

    # Normalize onto the z=1 plane in camera coords.
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    z = points_cam[:, 2]

    # Apply camera-model-specific distortion.
    if camera.model == "THIN_PRISM_FISHEYE":
        x, y = apply_thin_prism_fisheye_distortion(x, y, camera.distortion)
    # PINHOLE / SIMPLE_PINHOLE: no distortion -- use (x, y) directly.

    u = camera.fx * x + camera.cx
    v = camera.fy * y + camera.cy

    # Guard against any remaining non-finite outputs (extreme distortion
    # angles, accumulated FP error). Drop them before casting to int.
    finite = np.isfinite(u) & np.isfinite(v)
    u = u[finite]
    v = v[finite]
    z = z[finite].astype(np.float32)
    if u.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    u_int = np.floor(u).astype(np.int32)
    v_int = np.floor(v).astype(np.int32)
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_int = u_int[in_bounds]
    v_int = v_int[in_bounds]
    z = z[in_bounds]
    depth = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v_int, u_int), z)
    depth[~np.isfinite(depth)] = 0.0
    return depth


# ===========================================================================
# Scene discovery
# ===========================================================================
def discover_scenes(interim_dir: Path) -> dict[str, dict]:
    """Walk interim_dir and gather per-scene paths.

    Returns: {scene_name: {scene_dir, cameras_txt, images_txt, image_dir?,
                           scan_plys: [Path, ...], scan_alignment_mlp?}}
    """
    scenes: dict[str, dict] = {}

    # ETH3D's high-res multi-view training archive ships only the JPG
    # calibration variant (dslr_calibration_jpg/), which uses
    # THIN_PRISM_FISHEYE. No "undistorted" calibration is provided here.
    for cameras_txt in interim_dir.rglob("dslr_calibration_jpg/cameras.txt"):
        scene_dir = cameras_txt.parent.parent
        scene_name = scene_dir.name
        images_txt = cameras_txt.parent / "images.txt"
        if not images_txt.exists():
            continue
        scenes.setdefault(scene_name, {}).update({
            "scene_dir": scene_dir,
            "cameras_txt": cameras_txt,
            "images_txt": images_txt,
        })

    # DSLR images live under <scene>/images/dslr_images/*.JPG.
    for jpg in interim_dir.rglob("dslr_images/*.JPG"):
        for parent in jpg.parents:
            if parent.name in scenes:
                scenes[parent.name].setdefault("image_dir", jpg.parent)
                break
    # Fallback for any *.JPG, only if a scene still has no image_dir.
    for jpg in interim_dir.rglob("*.JPG"):
        for parent in jpg.parents:
            if parent.name in scenes:
                scenes[parent.name].setdefault("image_dir", jpg.parent)
                break

    # Scan point clouds: collect ALL scan*.ply (a scene is scanned from
    # multiple laser positions). Each gets its own alignment matrix later.
    for ply in interim_dir.rglob("*.ply"):
        for parent in ply.parents:
            if parent.name in scenes:
                scenes[parent.name].setdefault("scan_plys", []).append(ply)
                break

    # Alignment file
    for mlp in interim_dir.rglob("scan_alignment.mlp"):
        for parent in mlp.parents:
            if parent.name in scenes:
                scenes[parent.name]["scan_alignment_mlp"] = mlp
                break

    return scenes


def find_image_path(info: dict, pose_name: str) -> Path | None:
    """Resolve a DSLR image file given the COLMAP NAME field.

    The full pose name relative to scene_dir is tried first because it
    encodes the subfolder where the image lives. Falling back to
    image_dir / basename last avoids accidentally picking a same-named
    file in a different image folder.
    """
    name_path = Path(pose_name)
    candidates: list[Path] = []
    if "scene_dir" in info:
        scene_dir = info["scene_dir"]
        candidates.append(scene_dir / pose_name)
        candidates.append(scene_dir / "images" / pose_name)
        candidates.append(scene_dir / name_path)
    if "image_dir" in info:
        candidates.append(info["image_dir"] / name_path.name)
    for c in candidates:
        if c.exists():
            return c
    return None


# ===========================================================================
# Extraction
# ===========================================================================
def extract_if_needed(archive_path: Path, interim_dir: Path, marker_glob: str) -> None:
    """Extract a 7z into interim_dir unless any path matching marker_glob already exists."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    for _ in interim_dir.rglob(marker_glob):
        return  # already extracted
    print(f"  Extracting {archive_path.name} -> {interim_dir} (this can take several minutes)...")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=interim_dir)
    print(f"  {archive_path.name} extraction complete.")


# ===========================================================================
# Main pipeline
# ===========================================================================
def build_paths(root_dir: Path) -> dict[str, Path]:
    dataset_root = root_dir / "data" / "datasets" / "eth3d"
    return {
        "dataset_root": dataset_root,
        "jpg_archive": dataset_root / "raw" / JPG_ARCHIVE,
        "scan_archive": dataset_root / "raw" / SCAN_ARCHIVE,
        "interim_dir": dataset_root / "interim",
        "processed_dir": dataset_root / "processed",
        "depth_dir": dataset_root / "processed" / "depth",
        "samples_manifest": dataset_root / "processed" / "samples.jsonl",
        "stats_path": dataset_root / "processed" / "stats.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ETH3D + build dense GT depth maps.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract archives and recompute all depth maps from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    paths = build_paths(root_dir)

    for label, archive_path in (("jpg", paths["jpg_archive"]),
                                ("scan_eval", paths["scan_archive"])):
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Missing {label} archive: {archive_path}. "
                f"Run `python data\\scripts\\python\\acquire_remaining_benchmarks.py --benchmarks eth3d` first."
            )

    if args.force:
        shutil.rmtree(paths["interim_dir"], ignore_errors=True)
        shutil.rmtree(paths["processed_dir"], ignore_errors=True)

    # 1. Extract both archives (idempotent)
    extract_if_needed(paths["jpg_archive"], paths["interim_dir"], "cameras.txt")
    extract_if_needed(paths["scan_archive"], paths["interim_dir"], "*.ply")

    # 2. Discover scenes
    scenes = discover_scenes(paths["interim_dir"])
    if not scenes:
        raise RuntimeError(
            f"No scenes found under {paths['interim_dir']}. "
            f"Expected directories containing dslr_calibration_jpg/cameras.txt."
        )

    print(f"  Discovered {len(scenes)} scene(s):")
    for name in sorted(scenes):
        info = scenes[name]
        n_scans = len(info.get("scan_plys", []))
        ok = all(k in info for k in ("cameras_txt", "images_txt", "image_dir")) and n_scans > 0
        marker = "  OK " if ok else "  WARN"
        print(f"  {marker:6}  {name}  ({n_scans} scan(s))")

    # 3. Per-scene projection -> per-image depth.npy (resumable)
    paths["depth_dir"].mkdir(parents=True, exist_ok=True)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)

    sample_records: list[dict] = []
    images_per_scene: Counter[str] = Counter()
    skipped_scenes: list[str] = []
    skipped_images: list[str] = []

    for scene_name in sorted(scenes):
        info = scenes[scene_name]
        missing = [k for k in ("cameras_txt", "images_txt", "image_dir") if k not in info]
        if not info.get("scan_plys"):
            missing.append("scan_plys")
        if missing:
            skipped_scenes.append(f"{scene_name} (missing: {','.join(missing)})")
            continue

        cameras = parse_cameras_txt(info["cameras_txt"])
        poses = parse_images_txt(info["images_txt"])
        if not poses:
            skipped_scenes.append(f"{scene_name} (no poses)")
            continue

        # Resumability: if every output already exists, skip scan loading.
        depth_outs = {
            pose.image_id: paths["depth_dir"] / f"{scene_name}__{Path(pose.name).stem}.npy"
            for pose in poses
        }
        needs_projection = any(not depth_outs[p.image_id].exists() for p in poses)

        points = None
        if needs_projection:
            points = load_and_merge_scans(info, scene_name)
        else:
            print(f"  [{scene_name}] all {len(poses)} depth maps already on disk, skipping load.")

        for pose in poses:
            camera = cameras.get(pose.camera_id)
            if camera is None:
                skipped_images.append(f"{scene_name}/{pose.name} (unknown camera_id)")
                continue
            image_path = find_image_path(info, pose.name)
            if image_path is None:
                skipped_images.append(f"{scene_name}/{pose.name} (image not found)")
                continue
            depth_out = depth_outs[pose.image_id]
            if not depth_out.exists():
                if points is None:
                    points = load_and_merge_scans(info, scene_name)
                R = quaternion_to_rotation(pose.qvec)
                t = np.array(pose.tvec, dtype=np.float64)
                depth = project_scan_to_depth(points, R, t, camera)
                np.save(depth_out, depth)
            sample_records.append({
                "sample_id": f"{scene_name}__{Path(pose.name).stem}",
                "scene": scene_name,
                "camera_model": camera.model,
                "image_path": str(image_path),
                "depth_path": str(depth_out),
                "image_relpath": str(image_path.relative_to(paths["interim_dir"])).replace("\\", "/"),
                "depth_relpath": str(depth_out.relative_to(paths["dataset_root"])).replace("\\", "/"),
                "width": camera.width,
                "height": camera.height,
            })
            images_per_scene[scene_name] += 1

        print(f"  [{scene_name}] processed {images_per_scene[scene_name]} images.")
        points = None  # release memory before next scene

    # 4. Write manifest + stats
    with paths["samples_manifest"].open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "samples": len(sample_records),
        "scenes": len(images_per_scene),
        "images_per_scene": dict(images_per_scene),
        "skipped_scenes": skipped_scenes,
        "skipped_images_count": len(skipped_images),
        "skipped_images_examples": skipped_images[:5],
        "samples_manifest": str(paths["samples_manifest"]),
        "depth_dir": str(paths["depth_dir"]),
    }
    paths["stats_path"].write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()