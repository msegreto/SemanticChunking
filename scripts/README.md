# Scripts

Questa cartella contiene gli entrypoint CLI del progetto.

Il loro ruolo è fare da sottile livello di ingresso: leggono argomenti da linea di comando, caricano configurazioni e poi delegano la logica reale ai moduli sotto `src/`.

File presenti:
- `run_pipeline.py`: entrypoint principale. Legge `--config`, costruisce `ExperimentOrchestrator` da `src/pipelines/experiment_orchestrator.py` e lancia il flusso completo.
- `run_extrinsic_eval.py`: script separato per la sola valutazione extrinsic. Viene richiamato anche dall'orchestrator tramite subprocess quando `evaluation.extrinsic` è abilitato.
- `download_datasets.py`: utility per verificare o scaricare i dataset raw usando `DatasetFactory` e i processor definiti in `src/datasets/`.
- `download_qasper_hf_to_normalized.py`: converte QASPER da Hugging Face nel formato normalizzato interno (`documents.jsonl`, `queries.json`, `qrels`, `evidences`, `answers`).
- `download_techqa_hf_to_normalized.py`: converte TechQA da Hugging Face nel formato normalizzato interno, con estrazione best-effort di evidenze e risposte quando presenti.
- `__init__.py`: rende importabile il package `scripts` quando alcuni script vengono invocati come modulo.

Collegamenti principali:
- `run_pipeline.py` -> `src/pipelines/experiment_orchestrator.py`
- `run_extrinsic_eval.py` -> `src/evaluation/extrinsic/`
- `download_datasets.py` -> `src/datasets/`

Questa cartella non contiene business logic complessa: la logica importante resta nei moduli applicativi.

Note aggiornate:
- `run_extrinsic_eval.py` esegue i task in `evaluation.extrinsic_tasks_to_run` e, se un task non e' supportato o fallisce, lo marca come `skipped` senza interrompere gli altri.
- `download_datasets.py` supporta `--tasks` (lista separata da virgole) e controlli best-effort per artifact opzionali (`--evidence-path`, `--answers-path`) senza hard-fail.
