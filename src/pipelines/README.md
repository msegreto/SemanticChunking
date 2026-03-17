# Pipelines Module

Questa cartella contiene l'orchestrazione end-to-end del framework.

File presenti:
- `experiment_orchestrator.py`: cuore del flusso esecutivo. Carica la configurazione YAML, istanzia i componenti tramite le factory e coordina tutte le fasi del pipeline.
- `__init__.py`: abilita gli import del package.

Ruolo di `experiment_orchestrator.py`:
1. carica e valida implicitamente la configurazione;
2. chiama il dataset processor corretto tramite `DatasetFactory`;
3. esegue il flusso **streaming-only** a finestre (window);
4. per ogni finestra: split incrementale, routing (se attivo), chunking, intrinsic aggregata, embedding e indexing incrementali;
5. salva checkpoint (`stream_state.json` e `split_state.json`) per resume;
6. a streaming completato, invoca la valutazione extrinsic tramite `scripts/run_extrinsic_eval.py` (se abilitata).

Nota importante:
- il router nel flusso streaming opera oggi a livello di singolo documento (doc-by-doc);
- la struttura del codice include un placeholder `_run_router_window(...)` per future politiche di routing window-level.

Collegamenti con il progetto:
- entra dal lato CLI tramite `scripts/run_pipeline.py`;
- usa configurazioni in `configs/experiments/`;
- consuma e produce artefatti in `data/` e `results/`;
- dipende da quasi tutti i package sotto `src/`.
