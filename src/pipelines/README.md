# Pipelines Module

Questa cartella contiene l'orchestrazione end-to-end del framework.

File presenti:
- `experiment_orchestrator.py`: cuore del flusso esecutivo. Carica la configurazione YAML e coordina in sequenza i transformer della pipeline.
- `__init__.py`: abilita gli import del package.

Ruolo di `experiment_orchestrator.py`:
1. carica e valida implicitamente la configurazione;
2. prepara il dataset normalizzato;
3. esegue `split >> routing opzionale >> chunking`;
4. esegue intrinsic evaluation sul `chunk_output`;
5. esegue retrieval sul corpus chunked;
6. esegue extrinsic evaluation consumando direttamente l'output del retrieval.

Nota importante:
- il router e' opzionale e agisce sullo `split_output`;
- retrieval ed extrinsic condividono gli stessi artefatti (`manifest.json`, `items.jsonl`, `run.tsv`).

Collegamenti con il progetto:
- entra dal lato CLI tramite `scripts/run_pipeline.py`;
- usa configurazioni in `configs/experiments/`;
- consuma e produce artefatti in `data/` e `results/`;
- dipende da quasi tutti i package sotto `src/`.
