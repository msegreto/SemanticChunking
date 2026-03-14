# Pipelines Module

Questa cartella contiene l'orchestrazione end-to-end del framework.

File presenti:
- `experiment_orchestrator.py`: cuore del flusso esecutivo. Carica la configurazione YAML, istanzia i componenti tramite le factory e coordina tutte le fasi del pipeline.
- `__init__.py`: abilita gli import del package.

Ruolo di `experiment_orchestrator.py`:
1. carica e valida implicitamente la configurazione;
2. chiama il dataset processor corretto tramite `DatasetFactory`;
3. esegue lo splitter tramite `SplitterFactory`;
4. applica il router, se abilitato;
5. costruisce i chunk tramite `ChunkerFactory`;
6. esegue la valutazione intrinsic;
7. prova a riusare un indice retrieval gia' compatibile, se presente;
8. altrimenti produce embeddings e costruisce l'indice retrieval;
9. invoca la valutazione extrinsic tramite `scripts/run_extrinsic_eval.py`, se la fase e' abilitata.

Collegamenti con il progetto:
- entra dal lato CLI tramite `scripts/run_pipeline.py`;
- usa configurazioni in `configs/experiments/`;
- consuma e produce artefatti in `data/` e `results/`;
- dipende da quasi tutti i package sotto `src/`.
