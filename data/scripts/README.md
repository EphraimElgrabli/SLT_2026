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
python data/scripts/slt_data.py benchmarks --benchmarks kitti nyu_depth_v2
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

- `bootstrap`: clone the pinned upstream repository, create `data/.venv`, install requirements, and download the small checkpoint.
- `sanity-check`: run the example-image smoke test and write outputs into `data/outputs/sanity_check`.
- `setup`: run `bootstrap` and `sanity-check` together.
- `da2k`: download and preprocess the `DA-2K` benchmark.
- `benchmarks`: download the remaining paper benchmark archives.
