# Scripts

Questa cartella contiene gli entrypoint CLI del progetto.

Il loro ruolo è fare da sottile livello di ingresso: leggono argomenti da linea di comando, caricano configurazioni e poi delegano la logica reale ai moduli sotto `src/`.

File presenti:
- `run_pipeline.py`: entrypoint principale. Legge `--config`, costruisce `ExperimentOrchestrator` da `src/pipelines/experiment_orchestrator.py` e lancia il flusso completo.
- `run_split_normalized.py`: esegue solo lo split (sentence streaming) partendo dal dataset normalizzato e salva uno stato persistente su `state_split.json` (di default accanto a `split.output_path`).
- `run_split_all_normalized.py`: esegue lo split sentence su tutti i dataset già presenti sotto `data/normalized/**` (sia BEIR sia non-BEIR), creando output e `state_split.json` per ogni dataset.
- `precompute_split_embeddings.py`: pre-calcola embeddings per ogni riga dei file split (`sentences.jsonl`) e li salva dentro il JSONL (`embeddings.<model/cache_key>`), con stato `state_embeddings_<cache_key>.json`.
- `run_experiments_resume.py`: lancia in batch tutti i YAML di una cartella con stato persistente su JSON; se interrompi l'esecuzione, rilanciando riparte dagli esperimenti mancanti.
- `run_extrinsic_eval.py`: script separato per la sola valutazione extrinsic. Viene richiamato anche dall'orchestrator tramite subprocess quando `evaluation.extrinsic` è abilitato.
- `audit_extrinsic_coverage.py`: dato un dataset e una cartella YAML, fa audit completo (YAML/indici/tabelle estrinseche mancanti) e puo' lanciare il backfill delle evaluation extrinsic mancanti, poi rigenera le tabelle aggregate del dataset.
- `build_tables_graphs.py`: aggrega i risultati in tabelle/plot per extrinsic (`results/extrinsic/*.csv`) e intrinsic (`results/intrinsic/*.json`, escludendo `*_global.json`).
  Salva gli output in gerarchia `output_dir/<dataset>/<task>/{tables,plots}` (quindi prima dataset, poi task/metrica).
  Produce anche `global/metrics_long.csv` in formato tidy (dataset/task/method/metric/k/value/source) ordinato per dataset->task->metrica.
- `prewarm_normalized_datasets.py`: legge i config esperimenti, deduplica i dataset richiesti e prepara in anticipo i cache normalizzati sotto `data/normalized/` (o root custom), così i run successivi evitano i tempi di download/conversione.
- `download_datasets.py`: utility per verificare o scaricare i dataset raw usando `DatasetFactory` e i processor definiti in `src/datasets/`.
- `download_qasper_hf_to_normalized.py`: converte QASPER da Hugging Face nel formato normalizzato interno (`documents.jsonl`, `queries.json`, `qrels`, `evidences`, `answers`).
- `download_techqa_hf_to_normalized.py`: converte TechQA da Hugging Face nel formato normalizzato interno, con estrazione best-effort di evidenze e risposte quando presenti.
- `__init__.py`: rende importabile il package `scripts` quando alcuni script vengono invocati come modulo.

Collegamenti principali:
- `run_pipeline.py` -> `src/pipelines/experiment_orchestrator.py`
- `run_split_normalized.py` -> `src/config/loader.py`, `src/datasets/`, `src/splitting/`
- `run_split_all_normalized.py` -> `src/splitting/` + scansione diretta di `data/normalized/**/metadata.json`
- `precompute_split_embeddings.py` -> `src/embeddings/` + aggiornamento in-place dei JSONL di split
- `run_experiments_resume.py` -> `run_pipeline.py` (in loop sui config YAML)
- `run_extrinsic_eval.py` -> `src/evaluation/extrinsic/`
- `audit_extrinsic_coverage.py` -> `src/config/loader.py`, `scripts/run_extrinsic_eval.py`, `scripts/build_tables_graphs.py`
- `prewarm_normalized_datasets.py` -> `src/config/loader.py`, `src/datasets/`
- `download_datasets.py` -> `src/datasets/`

Questa cartella non contiene business logic complessa: la logica importante resta nei moduli applicativi.

Note aggiornate:
- `run_extrinsic_eval.py` esegue i task in `evaluation.extrinsic_tasks_to_run` e, se un task non e' supportato o fallisce, lo marca come `skipped` senza interrompere gli altri.
- `prewarm_normalized_datasets.py` unisce anche i task extrinsic richiesti dai vari YAML (`document_retrieval`, `evidence_retrieval`, `answer_generation`) per preparare cache coerenti prima dei run batch.
- `download_datasets.py` supporta `--tasks` (lista separata da virgole) e controlli best-effort per artifact opzionali (`--evidence-path`, `--answers-path`) senza hard-fail.
- `run_experiments_resume.py` non riesegue i config con stato `completed`; per rieseguire quelli `failed` usare `--retry-failed`.
- `audit_extrinsic_coverage.py` supporta `--dry-run` per solo audit senza calcoli; senza `--dry-run` tenta il backfill dei CSV extrinsic mancanti solo se trova indice + metadata validi.

Uso rapido batch resume:

```bash
python scripts/run_experiments_resume.py
```

Prewarm dei dataset normalizzati (download + conversione una volta sola):

```bash
python scripts/prewarm_normalized_datasets.py
```

Opzioni utili:
- `--configs-dir`: cartella dei YAML (default: `configs/experiments/grid_search_stage1_stella`)
- `--state-file`: file JSON di stato (default: `results/runner_state/grid_search_stage1_stella.json`)
- `--logs-dir`: log per-esperimento (default: `logs`)
- `--retry-failed`: include nel run anche i config marcati `failed`
- `--dry-run`: mostra coda e percorsi senza eseguire nulla

Output runtime:
- durante l'esecuzione l'output di ogni esperimento viene mostrato a terminale;
- lo stesso output viene salvato anche su file `logs/<nome_esperimento>.txt` (o nella cartella passata con `--logs-dir`).
- tutti i run forzano `CUDA_VISIBLE_DEVICES=0` (GPU 0).
