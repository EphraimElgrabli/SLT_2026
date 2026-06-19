# Scripts Workspace

This folder now uses a single Python-driven command surface so the workflows work on macOS, Linux, and Windows without depending on `bash`.

## Structure

- `slt_data.py`: master entry point for all supported workflows.
- `python/`: implementation modules for bootstrap, downloads, and preprocessing.
- `shell/`: Unix shell wrappers that forward to `slt_data.py`.
- `powershell/`: Windows PowerShell wrappers that forward to `slt_data.py`.

## Recommended Usage

The most portable option is the Python master script:

```bash
python data/scripts/slt_data.py --help
python data/scripts/slt_data.py setup
python data/scripts/slt_data.py da2k
python data/scripts/slt_data.py evaluate-da2k
python data/scripts/slt_data.py benchmarks --benchmarks kitti nyu_depth_v2
python data/scripts/slt_data.py metric-checkpoints
python data/scripts/slt_data.py evaluate-metric --datasets kitti nyu_depth_v2 sintel diode eth3d
```

## Wrapper Usage

Unix shell:

```bash
./data/scripts/shell/slt-data.sh setup
```

PowerShell:

```powershell
.\data\scripts\powershell\slt-data.ps1 setup
```

## Command Map

- `bootstrap`: clone the pinned upstream repository, create `data/.venv`, install requirements, and download the public student checkpoints.
- `sanity-check`: run the example-image smoke test and write outputs into `data/outputs/sanity_check`.
- `setup`: run `bootstrap` and `sanity-check` together.
- `da2k`: download and preprocess the `DA-2K` benchmark.
- `evaluate-da2k`: run the released `vits`, `vitb`, and `vitl` student checkpoints on processed DA-2K and write prediction/accuracy outputs.
- `benchmarks`: download the remaining paper benchmark archives.
- `metric-checkpoints`: download the small indoor/outdoor metric-depth checkpoints.
- `evaluate-metric`: run AbsRel, RMSE, and delta1 evaluation for supported local metric-depth benchmarks.

## DA-2K Report

After `evaluate-da2k` finishes, build the reproduction report with:

```bash
python data/scripts/python/build_da2k_report.py
```

This writes:

- `reports/da2k_reproduction.docx`
- `reports/da2k_artifact_hashes.txt`

## Tests

The fast unit tests cover command argument forwarding and do not require datasets or model checkpoints:

```bash
python -m unittest discover -s tests -v
```

The current completed stage includes the DA-2K relative-depth reproduction and the metric-depth evaluator for KITTI, NYU Depth V2, Sintel, DIODE, and ETH3D. ETH3D is evaluated against sparse COLMAP observations from the DSLR calibration archive, not dense projected scan maps.
