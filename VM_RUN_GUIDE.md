# Run on VM: Installation -> Execution

This guide summarizes the minimum steps required to run the project on a Linux VM.

## 1) VM Requirements

- Recommended OS: Ubuntu 22.04+.
- NVIDIA GPU available at index `0` (the code forces `CUDA_VISIBLE_DEVICES=0`).
- NVIDIA drivers installed and `nvidia-smi` working.
- Python `3.10`.
- `git`.

Quick check:

```bash
nvidia-smi
python3 --version
```

## 2) Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:
- `spacy` is included in the requirements; if the `en_core_web_sm` model is missing, the splitter automatically falls back to `spacy.blank("en")` + sentencizer.
- Some embeddings (e.g. `mpnet`) may require large downloads from Hugging Face.

## 3) (Optional but Recommended) Hugging Face Token

If you use models/datasets with rate limits or gated access:

```bash
export HF_TOKEN="<your_token>"
```

## 4) Prewarm Normalized Datasets (Recommended)

Prepare `data/normalized/...` in advance to reduce runtime and failures during runs:

```bash
python -m scripts.prewarm_normalized_datasets \
  --configs-dir configs/experiments/grid_search_stage1_mpnet
```

## 5) Run a Single Experiment

Example using a real config available in the repository:

```bash
python -m scripts.run_pipeline \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml
```

## 6) Batch Run with Automatic Resume

```bash
python -m scripts.run_experiments_resume \
  --configs-dir configs/experiments/grid_search_stage1_mpnet \
  --state-file results/runner_state/grid_search_stage1_mpnet.json \
  --logs-dir logs
```

If you want to retry failed runs:

```bash
python -m scripts.run_experiments_resume --retry-failed
```

## 7) Useful Scripts (Split + Precompute Embeddings)

### 7.1 Split-Only from YAML Config

Runs only the split step on the config's normalized dataset and saves `state_split.json`.

```bash
python -m scripts.run_split_normalized \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml
```

Dry run:

```bash
python -m scripts.run_split_normalized \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml \
  --dry-run
```

### 7.2 Split All Normalized Datasets

Scans `data/normalized/**` and generates splits under `data/processed/**`.

```bash
python -m scripts.run_split_all_normalized --continue-on-error
```

Dry run:

```bash
python -m scripts.run_split_all_normalized --dry-run
```

### 7.3 Precompute Embeddings on JSONL Splits

Precompute and store embeddings inside `sentences.jsonl` (to reuse in later runs):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model mpnet
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model bge
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model mpnet
```

Dry run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model mpnet --dry-run
```

## 8) Where to Find Outputs and Logs

- Per-experiment logs: `logs/*.txt`
- Batch resume state: `results/runner_state/*.json`
- Extrinsic results: `results/extrinsic/*.csv`
- Data/chunk/index artifacts: under `data/`

## 9) Docker Note (Important)

In the current repository, `docker-compose.yml` and `dockerfile` point to:

- `configs/experiments/base.yaml`

but that file is currently missing. If you use Docker, pass an existing config instead, for example:

```bash
python -u -m scripts.run_pipeline \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml
```

## 10) Common Issues

- `GPU 0 required` / `nvidia-smi not available` error:
  the VM does not meet the GPU requirements expected by the scripts.
- `ModuleNotFoundError`:
  virtual environment not active or dependencies not installed.
- Slow/failed HF downloads:
  set `HF_TOKEN` and try again.
