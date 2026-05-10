# ETH3D evaluation status

## Download status

The ETH3D assets requested by the benchmark downloader are present locally:

- `data/datasets/eth3d/raw/multi_view_training_dslr_jpg.7z`
- `data/datasets/eth3d/raw/multi_view_training_dslr_scan_eval.7z`

`data/datasets/eth3d/metadata.json` records the downloaded files and byte counts.

## Archive inspection

The downloaded ETH3D archives are not direct monocular RGB/depth pairs:

- `multi_view_training_dslr_jpg.7z` contains DSLR images and COLMAP-style calibration files such as `cameras.txt`, `images.txt`, and `points3D.txt`.
- `multi_view_training_dslr_scan_eval.7z` contains scan alignment files and large `.ply` scan point clouds.

## Implemented evaluation path

The current evaluator implements ETH3D as a sparse metric-depth benchmark:

1. Extract the DSLR image/calibration subset from `multi_view_training_dslr_jpg.7z` into `data/datasets/eth3d/interim/dslr_jpg/`.
2. Read COLMAP-style `cameras.txt`, `images.txt`, and `points3D.txt` files.
3. Convert registered 3D observations into camera-space metric depth at the observed 2D feature locations.
4. Evaluate Depth Anything V2 metric predictions at those sparse valid pixels.
5. Write AbsRel, RMSE, and delta1 outputs under `data/outputs/metric_depth/`.

The completed run evaluated 454 ETH3D DSLR images and wrote:

- `data/outputs/metric_depth/eth3d_details.json`
- `data/outputs/metric_depth/eth3d_records.jsonl`
- `data/outputs/metric_depth/summary.json`

## Caveat

This is not the same as a dense ETH3D scan-depth evaluation. A dense version would still need scan point-cloud loading, camera projection, occlusion handling, z-buffering, and per-image dense valid masks from `multi_view_training_dslr_scan_eval.7z`.
