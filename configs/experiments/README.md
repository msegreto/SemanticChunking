# Experiments Configs

Questa cartella contiene i file YAML che descrivono un esperimento end-to-end.

Ogni file viene passato a `scripts/run_pipeline.py`, che a sua volta carica `src/pipelines/experiment_orchestrator.py`. L'orchestrator legge le sezioni del YAML e attiva in sequenza dataset processing, splitting, routing, chunking, intrinsic evaluation, embedding, indexing ed extrinsic evaluation.

File presenti:
- `base.yaml`: configurazione baseline oggi più importante. Usa `beir/fiqa`, sentence splitting, router disabilitato, chunking `fixed`, embedder `all-mpnet-base-v2`, retriever `numpy` e valutazioni intrinsic + extrinsic.
- `msmarco_fixed_3.yaml`: variante sperimentale aggiuntiva, utile come secondo esempio di configurazione.

Come si collega al resto del progetto:
- sezione `dataset`: viene consumata dai processor in `src/datasets/`;
- sezione `split`: seleziona uno splitter in `src/splitting/`;
- sezione `router`: controlla il routing in `src/routing/`;
- sezione `chunking`: seleziona la strategia in `src/chunking/`;
- sezione `embedding`: sceglie il modello in `src/embeddings/`;
- sezione `retrieval`: guida la costruzione indice in `src/retrieval/`;
- sezione `evaluation`: guida `src/evaluation/intrinsic/` e `scripts/run_extrinsic_eval.py`.

In pratica, questa cartella rappresenta il punto di configurazione centrale del framework.
