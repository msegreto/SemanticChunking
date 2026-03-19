# Scripts

Questa cartella contiene gli entrypoint CLI del progetto.

Il loro ruolo è fare da sottile livello di ingresso: leggono argomenti da linea di comando, caricano configurazioni e poi delegano la logica reale ai moduli sotto `src/`.

File presenti:
- `run_pipeline.py`: entrypoint principale. Legge `--config`, costruisce `ExperimentOrchestrator` da `src/pipelines/experiment_orchestrator.py` e lancia il flusso completo.
- `run_experiments_resume.py`: lancia in batch tutti i YAML di una cartella con stato persistente su JSON; se interrompi l'esecuzione, rilanciando riparte dagli esperimenti mancanti.
- `run_extrinsic_eval.py`: script separato per la sola valutazione extrinsic. Viene richiamato anche dall'orchestrator tramite subprocess quando `evaluation.extrinsic` è abilitato.
- `download_datasets.py`: utility per verificare o scaricare i dataset raw usando `DatasetFactory` e i processor definiti in `src/datasets/`.
- `download_qasper_hf_to_normalized.py`: converte QASPER da Hugging Face nel formato normalizzato interno (`documents.jsonl`, `queries.json`, `qrels`, `evidences`, `answers`).
- `download_techqa_hf_to_normalized.py`: converte TechQA da Hugging Face nel formato normalizzato interno, con estrazione best-effort di evidenze e risposte quando presenti.
- `__init__.py`: rende importabile il package `scripts` quando alcuni script vengono invocati come modulo.

Collegamenti principali:
- `run_pipeline.py` -> `src/pipelines/experiment_orchestrator.py`
- `run_experiments_resume.py` -> `run_pipeline.py` (in loop sui config YAML)
- `run_extrinsic_eval.py` -> `src/evaluation/extrinsic/`
- `download_datasets.py` -> `src/datasets/`

Questa cartella non contiene business logic complessa: la logica importante resta nei moduli applicativi.

Note aggiornate:
- `run_extrinsic_eval.py` esegue i task in `evaluation.extrinsic_tasks_to_run` e, se un task non e' supportato o fallisce, lo marca come `skipped` senza interrompere gli altri.
- `download_datasets.py` supporta `--tasks` (lista separata da virgole) e controlli best-effort per artifact opzionali (`--evidence-path`, `--answers-path`) senza hard-fail.
- `run_experiments_resume.py` non riesegue i config con stato `completed`; per rieseguire quelli `failed` usare `--retry-failed`.

Uso rapido batch resume:

```bash
python scripts/run_experiments_resume.py
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
