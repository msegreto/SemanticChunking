# Chunking Module

Questa cartella implementa la fase di chunking come stage di pipeline. L'unica API pubblica e' [`ChunkTransformer`](/Users/mattiasegreto/Desktop/TesiCode/src/chunking/transformer.py), che costruisce internamente la strategia richiesta da `chunking.type`.

Struttura:
- `transformer.py`: entrypoint pubblico. Espone `ChunkTransformer.from_config(...)` e seleziona la strategia.
- `fixed.py`: implementazione interna per `chunking.type: fixed`.
- `semantic_breakpoint.py`: implementazione interna per `chunking.type: semantic_breakpoint`.
- `semantic_clustering.py`: implementazione interna per `chunking.type: semantic_clustering`.
- `_common.py`: helper interni condivisi per validazione, cache e persistenza.
- `semantic_base.py`: supporto interno comune ai metodi semantici.
- `semantic_utils.py`: utility interne per config semantica e ordinamento delle unita'.

Come si collega al resto del progetto:
- input: `split_units` prodotti da `src/splitting/` o passati attraverso `src/routing/`;
- output: lista di chunk usata da `src/evaluation/intrinsic/`, `src/embeddings/` e indirettamente da `src/retrieval/`;
- persistenza: le implementazioni interne possono leggere la cache per-run in `data/chunks/<dataset>/<yaml_name>/`; la scrittura su disco avviene solo con `save_chunks: true`.

Nel flusso attuale, il chunking viene eseguito come stage dopo lo split: `dataset >> split >> chunking`.

Naming consigliato per i metodi semantici:
- `chunking.type: semantic_breakpoint`
- `chunking.type: semantic_clustering`

Questo evita collisioni nella cache e mantiene leggibili i risultati sperimentali.

Config minima consigliata per `semantic_clustering`:
- `clustering_mode: single_linkage` oppure `dbscan`
- `lambda_weight: 0.5`
- `number_of_chunks: <int>` obbligatorio per `single_linkage`
- `stop_distance_threshold: 0.5` per `single_linkage`
- `dbscan_eps: 0.3` per `dbscan`
- `dbscan_min_samples: 2` per `dbscan`

Note implementative:
- il metodo e' sentence-level come nel paper;
- i chunk possono contenere frasi non contigue;
- il testo finale del chunk concatena comunque le frasi nell'ordine originale del documento;
- per `single_linkage`, `max_chunk_size` viene derivato come `ceil(num_sentences / number_of_chunks)` per ciascun documento;
- per `dbscan` il backend usato e' `scikit-learn` con matrice di distanza precomputata;
- `semantic_breakpoint` supporta `threshold_type` compatibili con LangChain, inclusi `percentile`, `standard_deviation`, `interquartile`, `gradient`, `distance` e `gradient_absolute`;
- i metodi semantici riusano l'embedder configurato in `chunking.embedding_model`, che l'orchestrator popola di default dalla sezione `embedding`.
