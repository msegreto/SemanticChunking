# Run Su VM: Installazione -> Esecuzione

Questa guida riassume i passaggi minimi per eseguire il progetto su una VM Linux.

## 1) Prerequisiti VM

- OS consigliato: Ubuntu 22.04+.
- GPU NVIDIA disponibile come indice `0` (il codice forza `CUDA_VISIBLE_DEVICES=0`).
- Driver NVIDIA installati e comando `nvidia-smi` funzionante.
- Python `3.10`.
- `git`.

Check rapido:

```bash
nvidia-smi
python3 --version
```



## 2) rea ambiente virtuale e installa dipendenze

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Note:
- `spacy` e' incluso nei requirements; se il modello `en_core_web_sm` non e' presente, lo splitter fa fallback automatico a `spacy.blank("en")` + sentencizer.
- Alcuni embedding (es. `stella`) possono richiedere download pesanti da Hugging Face.

## 3) (Opzionale ma consigliato) Token Hugging Face

Se usi modelli/dataset con rate-limit o gated access:

```bash
export HF_TOKEN="<il_tuo_token>"
```

## 4) Prewarm dataset normalizzati (consigliato)

Prepara in anticipo `data/normalized/...` per ridurre tempi/fail durante i run:

```bash
python -m scripts.prewarm_normalized_datasets \
  --configs-dir configs/experiments/grid_search_stage1_stella
```

## 5) Run di un singolo esperimento

Esempio con un config reale presente nel repo:

```bash
python -m scripts.run_pipeline \
  --config configs/experiments/grid_search_stage1_stella/fiqa_FIXED_01.yaml
```

## 6) Run batch con resume automatico

```bash
python -m scripts.run_experiments_resume \
  --configs-dir configs/experiments/grid_search_stage1_stella \
  --state-file results/runner_state/grid_search_stage1_stella.json \
  --logs-dir logs
```

Se vuoi ritentare quelli falliti:

```bash
python -m scripts.run_experiments_resume --retry-failed
```

## 7) Nuovi script utili (split + precompute embeddings)

### 7.1 Split-only da config YAML

Esegue solo lo split sul dataset normalizzato del config e salva `state_split.json`.

```bash
python -m scripts.run_split_normalized \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml
```

Dry-run:

```bash
python -m scripts.run_split_normalized \
  --config configs/experiments/grid_search_stage1_mpnet/fiqa_FIXED_01.yaml \
  --dry-run
```

### 7.2 Split su tutti i dataset normalizzati

Scansiona `data/normalized/**` e genera split sotto `data/processed/**`.

```bash
python -m scripts.run_split_all_normalized --continue-on-error
```

Dry-run:

```bash
python -m scripts.run_split_all_normalized --dry-run
```

### 7.3 Precompute embeddings sugli split JSONL

Per precomputare e salvare embeddings dentro i `sentences.jsonl` (riuso nei run successivi):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model mpnet
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model bge
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model stella
```

Dry-run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.precompute_split_embeddings --model mpnet --dry-run
```

## 8) Dove trovi output e log

- Log per esperimento: `logs/*.txt`
- Stato batch resume: `results/runner_state/*.json`
- Risultati extrinsic: `results/extrinsic/*.csv`
- Artefatti dati/chunk/index: sotto `data/`

## 9) Nota Docker (importante)

Nel repo attuale `docker-compose.yml` e `dockerfile` puntano a:

- `configs/experiments/base.yaml`

ma quel file oggi non e' presente. Se usi Docker, devi passare un config esistente, ad esempio:

```bash
python -u -m scripts.run_pipeline \
  --config configs/experiments/grid_search_stage1_stella/fiqa_FIXED_01.yaml
```

## 10) Problemi comuni

- Errore `GPU 0 required` / `nvidia-smi not available`:
  la VM non soddisfa i prerequisiti GPU richiesti dagli script.
- `ModuleNotFoundError`:
  ambiente virtuale non attivo o dipendenze non installate.
- Download lenti/falliti da HF:
  imposta `HF_TOKEN` e riprova.
