# Experiments Configs

Questa cartella contiene i file YAML che descrivono un esperimento end-to-end.

Ogni file viene passato a `scripts/run_pipeline.py`, che a sua volta carica `src/pipelines/experiment_orchestrator.py`. L'orchestrator legge le sezioni del YAML e attiva in sequenza dataset processing, splitting, routing, chunking, intrinsic evaluation, retrieval ed extrinsic evaluation.

## Aggiornamento stato attuale

Il pipeline e' oggi lineare e stage-based:
- `dataset >> split >> router opzionale >> chunking >> intrinsic >> retrieval >> extrinsic`;
- il riuso avviene tramite artefatti salvati dalle singole fasi;
- non esistono piu' finestre streaming o resume incrementale nell'orchestrator.

File presenti:
- `base.yaml`: configurazione baseline oggi piu' importante. Usa `beir/fiqa`, sentence splitting, router disabilitato, chunking `fixed`, embedder logico `mpnet`, retrieval dense FAISS e valutazioni intrinsic + extrinsic.
- `msmarco_fixed_3.yaml`: variante sperimentale aggiuntiva, utile come secondo esempio di configurazione.

Come si collega al resto del progetto:
- sezione `dataset`: viene consumata dai processor in `src/datasets/`;
- sezione `split`: seleziona uno splitter in `src/splitting/`;
- sezione `router`: controlla il routing in `src/routing/`;
- sezione `chunking`: seleziona la strategia in `src/chunking/`; per i metodi semantici in riproduzione usare type espliciti come `semantic_breakpoint` e `semantic_clustering`, non `semantic`;
- sezione `embedding`: sceglie il modello in `src/embeddings/`;
- sezione `retrieval`: guida la costruzione indice in `src/retrieval/`;
- sezione `evaluation`: guida `src/evaluation/intrinsic/` e `src/evaluation/extrinsic/`.

In pratica, questa cartella rappresenta il punto di configurazione centrale del framework.

Nota sul refactor config: la sezione `evaluation` e' stata centralizzata in profili sotto `configs/evaluations/`.
Nei file in `configs/experiments/` e' consigliato usare:
- `evaluation_profile: <nome_profilo>`
- un blocco `evaluation:` minimale solo con gli override (es. `extrinsic_tasks_to_run`, `intrinsic_model.backend`, `ks` specifici).

Anche la sezione `execution` puo' essere centralizzata in profili sotto `configs/execution/`.
Nei file esperimento e' consigliato usare:
- `execution_profile: <nome_profilo>` (es. `default` o `force_rebuild`)
- un blocco `execution:` minimale solo con override locali per `allow_reuse`, `force_rebuild` o `stages.<nome_fase>`.

## Come costruire un YAML completo

Quando crei un nuovo file in questa cartella, il modo piu' semplice e sicuro e' partire da uno YAML esistente e modificare solo le parti necessarie. In generale:

1. scegli il dataset;
2. scegli lo split (`sentence` oggi e' quello realmente usato);
3. scegli se abilitare il router;
4. scegli il chunker e i suoi parametri specifici;
5. scegli embedder e retrieval model;
6. decidi quali valutazioni attivare;
7. decidi la politica di riuso cache tramite `execution`.

Template completo consigliato:

```yaml
experiment_name: "nome_esperimento"
evaluation_profile: default
execution_profile: default

execution:
  stages:
    split:
      force_rebuild: true

dataset:
  name: "beir/fiqa"            # oppure "msmarco" / "msmarco-docs"
  normalized_dir: "data/normalized/beir/fiqa"   # opzionale
  split: "dev"
  download_if_missing: true
  delete_raw_after_normalized: true             # opzionale
  force_rebuild_normalized: false               # opzionale
  max_documents: null                           # opzionale, utile per debug
  queries_path: null                            # opzionale, fallback per extrinsic
  qrels_path: null                              # opzionale, fallback per extrinsic

split:
  type: sentence
  model: en_core_web_sm
  output_path: data/processed/fiqa/sentences.jsonl

router:
  enabled: false
  name: "moc"                  # alias supportati oggi: "moc", "default"

chunking:
  type: fixed                  # oppure semantic_breakpoint / semantic_clustering
  dataset: fiqa
  save_chunks: true

  # solo per fixed
  n_chunks: 3
  overlap_sentences: 0

  # solo per semantic chunking
  embedding_model: "mpnet"     # se omesso viene preso da embedding.name
  embedding_batch_size: 64
  similarity_metric: cosine
  show_embedding_progress: true

  # solo per semantic_breakpoint
  threshold_type: percentile
  threshold_value: 95
  min_chunk_size: 200
  number_of_chunks: null
  embedding_model_kwargs: {}
  embedding_encode_kwargs: {}
  sentence_split_regex: "(?<=[.?!])\\s+"

  # solo per semantic_clustering
  clustering_mode: single_linkage
  lambda: 0.5
  number_of_chunks: 4
  stop_distance_threshold: 0.5
  allow_non_contiguous_chunks: true
  dbscan_eps: 0.3
  dbscan_min_samples: 2

embedding:
  name: "mpnet"
  batch_size: 32
  normalize_embeddings: true
  show_progress_bar: true
  convert_to_numpy: true       # opzionale
  log_embedding_calls: true    # opzionale

retrieval:
  enabled: true
  backend: dense_faiss
  distance: cosine
  normalize: true
  output_dir: null             # opzionale

evaluation:
  extrinsic_tasks_to_run: [document_retrieval]
  intrinsic_model:
    backend: lexical           # override opzionale del profilo
  extrinsic_tasks:
    answer_generation:
      reader_backend: seq2seqlm
      reader_model: google/flan-t5-base
      top_k: 5
```

## Spiegazione delle sezioni

### `experiment_name`

Nome logico dell'esperimento. Viene usato nei nomi dei risultati extrinsic e come fallback in alcuni punti dell'orchestrator.

### `execution`

Controlla il riuso delle cache e i rebuild forzati.
Di solito viene risolta via `execution_profile` con eventuali override locali.

- `allow_reuse`: abilita il riuso globale quando possibile.
- `force_rebuild`: forza il ricalcolo globale di tutte le fasi che supportano cache.
- `stages.<nome_fase>.allow_reuse`: override per singola fase.
- `stages.<nome_fase>.force_rebuild`: override per singola fase.

Le fasi oggi gestite qui sono `dataset`, `split`, `chunking` e `retrieval`.

Profili pronti:
- `execution_profile: default`
- `execution_profile: force_rebuild`

### `dataset`

Definisce quale dataset usare e come gestire raw e normalized.

Campi piu' importanti:
- `name`: richiesto. Formati supportati oggi: `beir/<dataset_name>`, `msmarco`, `msmarco-docs`.
  - nota: `beir/qasper` e `beir/techqa` hanno un ingest dedicato che può costruire automaticamente la `normalized` tramite script converter, senza richiedere pre-download manuale.
- `normalized_dir`: opzionale; se omesso il codice usa `data/normalized/<dataset_name>`.
- `split`: split del dataset, per esempio `dev` o `test`.
- `download_if_missing`: se `true`, tenta il download automatico.
- `delete_raw_after_normalized`: di default il codice puo' cancellare la raw dopo il salvataggio corretto della normalized.
- `max_documents`: utile per run di debug; quando e' impostato, il codice evita il riuso della normalized completa.

Per la valutazione extrinsic esistono anche i fallback:
- `queries_path`
- `qrels_path`

Servono solo se non vuoi usare la risoluzione automatica su normalized dataset (`evaluation.extrinsic_tasks.document_retrieval.normalized_dataset_dir` o `dataset.normalized_dir`/`dataset.name`).

### `split`

Seleziona come trasformare i documenti in unita' testuali.

Campi piu' usati:
- `type`: `sentence` oppure `contextualizer`.
- `model`: modello spaCy per `sentence`.
- `output_path`: path del JSONL prodotto.

Nota: nei file attuali compare anche `split.split:` vuoto. Non sembra necessario per il codice attuale, quindi nei nuovi YAML puoi tranquillamente ometterlo.

### `router`

Attiva o disattiva la fase di routing.

- `enabled`: se `false`, il router viene saltato.
- `name`: oggi gli alias registrati sono `moc` e `default`, entrambi mappati al router dummy.

### `chunking`

Questa e' la sezione con piu' varianti, quindi conviene pensarla in due livelli:
- campi comuni sempre presenti;
- campi specifici del metodo scelto in `type`.

Campi comuni:
- `type`: `fixed`, `semantic_breakpoint` oppure `semantic_clustering`.
- `dataset`: nome corto usato per i path di cache chunk.
- `save_chunks`: se `true` salva la cache dei chunk in `data/chunks/...`.

Campi comuni al semantic chunking:
- `embedding_model`: embedder usato per embeddare le split units; se omesso, l'orchestrator copia `embedding.name`.
- `embedding_batch_size`: batch size per questa fase.
- `similarity_metric`: oggi il codice supporta solo `cosine`.
- `show_embedding_progress`: mostra la progress bar in fase di semantic chunking.

Per `type: fixed`:
- `n_chunks`: richiesto.
- `overlap_sentences`: richiesto dal chunker fixed.

Per `type: semantic_breakpoint`:
- `threshold_type`: supporta `percentile`, `standard_deviation`, `interquartile`, `gradient`, `distance`, `gradient_absolute`.
- `threshold_value`: opzionale; se omesso usa il default del metodo.
- `min_chunk_size`: opzionale.
- `number_of_chunks`: opzionale.
- `embedding_model_kwargs`: opzionale.
- `embedding_encode_kwargs`: opzionale.
- `sentence_split_regex`: opzionale.

Per `type: semantic_clustering`:
- `clustering_mode`: `single_linkage` oppure `dbscan`.
- `lambda`: peso della componente posizionale, in `[0, 1]`.
- `number_of_chunks`: obbligatorio per `single_linkage`.
- `stop_distance_threshold`: usato da `single_linkage`.
- `allow_non_contiguous_chunks`: booleano.
- `dbscan_eps`: usato da `dbscan`.
- `dbscan_min_samples`: usato da `dbscan`.

Nota importante: il codice accetta sia `lambda` sia `lambda_weight`, ma nei file YAML conviene usare `lambda` per restare coerenti con gli esempi gia' presenti.

### `embedding`

Definisce l'embedder condiviso usato dalle fasi che ne hanno bisogno, soprattutto semantic chunking e alcune metriche o evaluator.

Campi principali:
- `name`: oggi la factory supporta alias di `mpnet`, `bge` e `stella`.
- `batch_size`
- `normalize_embeddings`
- `show_progress_bar`

Campi opzionali gestiti dal validatore:
- `convert_to_numpy`
- `log_embedding_calls`

Nota pratica: con `retrieval.backend: dense_faiss` l'embedder viene usato anche dal retrieval. Resta inoltre rilevante per semantic chunking e per alcuni evaluator extrinsic.

### `retrieval`

Configura la costruzione dell'indice e del run di retrieval.

Campi principali:
- `enabled`
- `backend`: `dense_faiss`
- `distance`: per esempio `cosine` o `l2`
- `normalize`: normalizza gli embedding prima dell'indicizzazione
- `output_dir`: opzionale; se omesso usa `data/pt_indexes/<dataset>/<chunking>/<config_file_stem>`

Il retrieval salva indice, manifest, `items.jsonl` e, se il dataset espone i topics, anche `run.tsv`.

### `evaluation.extrinsic_tasks.answer_generation`

Quando abiliti `answer_generation`, il task usa un reader `pyterrier_rag` sui top chunk del retrieval.

Campi principali:
- `reader_backend`: oggi supportato `seq2seqlm`.
- `reader_model`: modello Hugging Face del reader, per esempio `google/flan-t5-base`.
- `top_k`: quanti chunk usare per generare la risposta.

### `evaluation`

Contiene sia intrinsic sia extrinsic, ma in modo tipico viene risolta come merge:
- profilo da `evaluation_profile` (es. `configs/evaluations/default.yaml`)
- override locali in `evaluation` nel file esperimento.

Override intrinsic piu' comuni nel file esperimento:
- `intrinsic_model.backend` (`lexical` oppure `hf_causal_lm`)
- `intrinsic_metrics.*` (se vuoi attivare/disattivare metriche)

Override extrinsic piu' comuni nel file esperimento:
- `extrinsic_tasks_to_run`
- `extrinsic_tasks.<task>.ks`
- opzionalmente `extrinsic_tasks.document_retrieval.normalized_dataset_dir` (per dataset con layout custom)
- opzionalmente `extrinsic_tasks.document_retrieval.query_subset_size` (es. `100`)
- opzionalmente `extrinsic_tasks.document_retrieval.query_subset_seed` (default `42`)
- opzionalmente `extrinsic_tasks.document_retrieval.include_full_query_set` (default `true`)

Se non specifichi path espliciti per evidences/answers, vengono risolti automaticamente a partire dalla normalized dir.

Esempio (document retrieval full + subset da 100 query):

```yaml
evaluation:
  extrinsic_tasks_to_run: [document_retrieval]
  extrinsic_tasks:
    document_retrieval:
      ks: [1, 3, 5, 10]
      query_subset_size: 100
      query_subset_seed: 42
      include_full_query_set: true
```

## Template minimi per i casi piu' comuni

### 1. Baseline `fixed`

Usa questo quando vuoi il setup piu' semplice e stabile:

```yaml
experiment_name: "fiqa_fixed"

dataset:
  name: "beir/fiqa"
  split: "dev"
  download_if_missing: true

split:
  type: sentence
  model: en_core_web_sm
  output_path: data/processed/fiqa/sentences.jsonl

router:
  enabled: false
  name: "moc"

chunking:
  type: fixed
  dataset: fiqa
  n_chunks: 3
  overlap_sentences: 0
  save_chunks: true

embedding:
  name: "mpnet"
  batch_size: 32
  normalize_embeddings: true
  show_progress_bar: true

retrieval:
  enabled: true
  backend: dense_faiss
  distance: cosine
  normalize: true

evaluation:
  extrinsic_tasks_to_run: [document_retrieval]
```

### 2. `semantic_breakpoint`

Usa questo se vuoi il chunking semantico con soglia sui breakpoint:

```yaml
chunking:
  type: semantic_breakpoint
  dataset: fiqa
  save_chunks: false
  embedding_model: "mpnet"
  embedding_batch_size: 64
  similarity_metric: cosine
  show_embedding_progress: true
  threshold_type: percentile
  threshold_value: 95
```

### 3. `semantic_clustering`

Usa questo se vuoi clustering semantico:

```yaml
chunking:
  type: semantic_clustering
  dataset: fiqa
  save_chunks: false
  embedding_model: "mpnet"
  embedding_batch_size: 64
  similarity_metric: cosine
  show_embedding_progress: true
  clustering_mode: single_linkage
  lambda: 0.5
  number_of_chunks: 4
  stop_distance_threshold: 0.5
  allow_non_contiguous_chunks: true
```

## Checklist finale prima di lanciare

- `dataset.name` e `chunking.dataset` sono coerenti tra loro.
- Se usi `document_retrieval`, query e documenti vengono codificati con lo stesso embedder configurato in `embedding.name`.
- Se usi `semantic_clustering` con `single_linkage`, hai messo `number_of_chunks`.
- Se usi `normalized_dataset_dir`, il path punta davvero alla cartella con `queries.json`, `qrels/` e `metadata.json`.
- Se vuoi evitare riuso di cache vecchie, imposta `execution.force_rebuild: true` oppure l'override di stage.
- Se vuoi risultati intrinsic con nomi leggibili, valorizza `evaluation.save.yaml_name`.
