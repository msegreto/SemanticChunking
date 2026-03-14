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
- `semantic.py`: alias intenzionalmente ambiguo che fallisce con un messaggio guidato; serve a forzare l'uso di nomi espliciti.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- input: `split_units` prodotti da `src/splitting/` o passati attraverso `src/routing/`;
- output: lista di chunk usata da `src/evaluation/intrinsic/`, `src/embeddings/` e indirettamente da `src/retrieval/`;
- persistenza: `fixed.py` usa `data/chunks/<dataset>/<yaml_name>/`; se i file della run esistono già e sono compatibili, li riusa anche con `save_chunks: false`.

Nel run baseline attuale, questa cartella è cruciale perché `fixed.py` genera i chunk su cui si basano tutte le fasi successive.

Naming consigliato per i metodi semantici:
- `chunking.type: semantic_breakpoint`
- `chunking.type: semantic_clustering`

Questo evita collisioni nella cache e mantiene leggibili i risultati sperimentali.

Config minima consigliata per `semantic_clustering`:
- `clustering_mode: single_linkage` oppure `dbscan`
- `lambda: 0.5`
- `number_of_chunks: <int>` obbligatorio per `single_linkage`
- `stop_distance_threshold: 0.5` per `single_linkage`
- `dbscan_eps: 0.3` per `dbscan`
- `dbscan_min_samples: 2` per `dbscan`

Note implementative:
- il metodo e' sentence-level come nel paper;
- i chunk possono contenere frasi non contigue;
- il testo finale del chunk concatena comunque le frasi nell'ordine originale del documento;
- per `single_linkage`, `max_chunk_size` viene derivato come `ceil(num_sentences / number_of_chunks)` per ciascun documento;
- per `dbscan` il backend usato e' `scikit-learn` con matrice di distanza precomputata.
