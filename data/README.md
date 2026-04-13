# Data Workspace

This folder contains the local workspace for the `Depth Anything V2` reproduction.

## Structure

- `scripts/`: Cross-platform workflow entry points and implementation modules.
- `models/`: Downloaded checkpoints.
- `datasets/`: Evaluation datasets and benchmark assets.
- `outputs/`: Generated predictions and sanity-check outputs.
- `external/`: Cloned third-party repositories. Generated locally and ignored by git.

## First Step

Run the bootstrap and smoke test through the master script:

```bash
python data/scripts/slt_data.py setup
```

The bootstrap clones the official repository at a pinned commit, creates a local virtual environment in `data/.venv`, installs upstream requirements, and downloads the small checkpoint used for an initial smoke test.

If you prefer OS-specific wrappers, use:

- Unix shell: `./data/scripts/shell/slt-data.sh setup`
- PowerShell: `.\data\scripts\powershell\slt-data.ps1 setup`

## DA-2K Dataset Workflow

Acquire and preprocess the first benchmark dataset:

```bash
python data/scripts/slt_data.py da2k
```

This workflow:

- downloads the official `DA-2K.zip` archive into `data/datasets/da2k/raw/`
- extracts the dataset into `data/datasets/da2k/interim/`
- validates the annotation file against the extracted images
- writes processed manifests and summary statistics into `data/datasets/da2k/processed/`

## Remaining Benchmark Downloads

Download the other evaluation benchmarks:

```bash
python data/scripts/slt_data.py benchmarks
```

This workflow downloads the official benchmark archives into dedicated folders under `data/datasets/`:

- `kitti/raw/`
- `nyu_depth_v2/raw/`
- `sintel/raw/`
- `eth3d/raw/`
- `diode/raw/`

The script also writes a `metadata.json` file for each benchmark with the source URLs and expected archive sizes.

Before running the full download set, make sure the machine has enough free disk space for the remaining archives. The downloader performs a free-space check and stops before starting a file that does not fit.

For more examples and the new folder layout, see [`data/scripts/README.md`](./scripts/README.md).
