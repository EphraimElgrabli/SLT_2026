# Data Workspace

This folder contains the local workspace for the `Depth Anything V2` reproduction.

## Structure

- `scripts/`: Small reproducible setup and validation scripts.
- `models/`: Downloaded checkpoints.
- `datasets/`: Evaluation datasets and benchmark assets.
- `outputs/`: Generated predictions and sanity-check outputs.
- `external/`: Cloned third-party repositories. Generated locally and ignored by git.

## First Step

Run the bootstrap:

```bash
./data/scripts/bootstrap_depth_anything_v2.sh
```

Then run the sanity check:

```bash
./data/scripts/run_depth_anything_v2_sanity_check.sh
```

The bootstrap clones the official repository at a pinned commit, creates a local virtual environment in `data/.venv`, installs upstream requirements, and downloads the small checkpoint used for an initial smoke test.

## DA-2K Dataset Workflow

Acquire and preprocess the first benchmark dataset:

```bash
./data/scripts/bootstrap_da2k_dataset.sh
```

This workflow:

- downloads the official `DA-2K.zip` archive into `data/datasets/da2k/raw/`
- extracts the dataset into `data/datasets/da2k/interim/`
- validates the annotation file against the extracted images
- writes processed manifests and summary statistics into `data/datasets/da2k/processed/`
