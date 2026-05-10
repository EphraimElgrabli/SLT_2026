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

The bootstrap clones the official repository at a pinned commit, creates a local virtual environment in `data/.venv`, installs upstream requirements, and downloads the public Depth Anything V2 student checkpoints used by the DA-2K reproduction (`vits`, `vitb`, and `vitl`).

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

## DA-2K Evaluation Workflow

Run the released student models on DA-2K and build the reproduction report:

```bash
python data/scripts/slt_data.py evaluate-da2k
python data/scripts/python/build_da2k_report.py
```

The evaluation workflow:

- runs `vits`, `vitb`, and `vitl` on the processed DA-2K images
- reads the predicted inverse-depth value at each annotated pixel pair
- marks a pair as correct when the model assigns higher inverse depth to the annotated closer point
- writes resumable prediction files under `data/outputs/da2k/predictions/`
- writes aggregate accuracy to `data/outputs/da2k/summary.json`

The report workflow writes:

- `reports/da2k_reproduction.docx`
- `reports/da2k_artifact_hashes.txt`

Current milestone status: DA-2K is acquired, preprocessed, evaluated, and reported. KITTI and NYU Depth V2 are acquired locally for the next stage, but their metric-depth preprocessing and evaluation are not part of this completed DA-2K milestone.

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

## Metric-Depth Evaluation Workflow

The metric stage uses the public small Depth Anything V2 metric checkpoints: Hypersim for indoor data and Virtual KITTI for outdoor data.

```bash
python data/scripts/slt_data.py metric-checkpoints
python data/scripts/slt_data.py evaluate-metric --datasets kitti nyu_depth_v2 sintel diode eth3d
```

The evaluator writes:

- `data/outputs/metric_depth/summary.json`
- `data/outputs/metric_depth/*_details.json`
- `data/outputs/metric_depth/*_records.jsonl`

The current adapters cover KITTI, NYU Depth V2, Sintel, DIODE, and ETH3D. ETH3D uses sparse COLMAP observations from the DSLR calibration archive as metric ground truth; dense ETH3D scan projection remains a separate future extension.
