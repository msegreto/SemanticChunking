# Scripts

Questa cartella contiene gli entrypoint CLI del progetto.

Il loro ruolo è fare da sottile livello di ingresso: leggono argomenti da linea di comando, caricano configurazioni e poi delegano la logica reale ai moduli sotto `src/`.

File presenti:
- `run_pipeline.py`: entrypoint principale. Legge `--config`, costruisce `ExperimentOrchestrator` da `src/pipelines/experiment_orchestrator.py` e lancia il flusso completo.
- `run_extrinsic_eval.py`: script separato per la sola valutazione extrinsic. Viene richiamato anche dall'orchestrator tramite subprocess quando `evaluation.extrinsic` è abilitato.
- `download_datasets.py`: utility per verificare o scaricare i dataset raw usando `DatasetFactory` e i processor definiti in `src/datasets/`.
- `__init__.py`: rende importabile il package `scripts` quando alcuni script vengono invocati come modulo.

Collegamenti principali:
- `run_pipeline.py` -> `src/pipelines/experiment_orchestrator.py`
- `run_extrinsic_eval.py` -> `src/evaluation/extrinsic/`
- `download_datasets.py` -> `src/datasets/`

Questa cartella non contiene business logic complessa: la logica importante resta nei moduli applicativi.
