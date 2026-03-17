# Chunking Module

Questa cartella implementa le strategie di chunking, cioè la trasformazione delle unità prodotte dallo splitting in chunk finali da usare per embedding, retrieval e valutazione.

File presenti:
- `base.py`: interfaccia astratta `BaseChunker`.
- `factory.py`: `ChunkerFactory`, usata dall'orchestrator per selezionare la strategia richiesta nel YAML.
- `fixed.py`: implementazione concreta oggi realmente funzionante. Raggruppa le sentence per documento, le divide in un numero prefissato di chunk, valida l'input e può riusare chunk già persistiti per la stessa run YAML.
- `semantic_base.py`: base class condivisa per i metodi semantici, con validazione e metadati comuni.
- `semantic_utils.py`: utility comuni per config semantica e ordinamento delle unita'.
- `semantic_breakpoint.py`: implementazione del metodo semantic breakpoint tramite backend LangChain, adattata all'output del framework.
- `semantic_clustering.py`: implementazione del metodo semantic clustering del paper, con supporto `single_linkage` e `dbscan`.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- input: `split_units` prodotti da `src/splitting/` o passati attraverso `src/routing/`;
- output: lista di chunk usata da `src/evaluation/intrinsic/`, `src/embeddings/` e indirettamente da `src/retrieval/`;
- persistenza: `fixed.py` usa `data/chunks/<dataset>/<yaml_name>/`; se i file della run esistono già e sono compatibili, li riusa anche con `save_chunks: false`.
- persistenza: tutti i chunker che estendono `BaseChunker` possono leggere la cache per-run in `data/chunks/<dataset>/<yaml_name>/`; la scrittura su disco avviene solo con `save_chunks: true`.

Nel flusso attuale (streaming-only), il chunking viene eseguito incrementale per window e per documento, mantenendo comunque invariati formato output e cache per-run.

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
