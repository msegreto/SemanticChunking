# Extrinsic Evaluation

Questa cartella contiene la valutazione extrinsic, cioè la misurazione del comportamento dei chunk su un task downstream reale.

Nel progetto attuale il task implementato è soprattutto il document retrieval.

File presenti:
- `base.py`: interfaccia `BaseExtrinsicEvaluator`.
- `factory.py`: factory per scegliere l'evaluator richiesto.
- `document_retrieval.py`: evaluator principale. Carica query e qrels, ricarica embedder e retriever, esegue la ricerca e calcola macro precision, recall e F1 a vari `k`.
- `io.py`: risolve i path a query, qrels e metadata indice, e carica questi file dal disco.
- `metrics.py`: metrica elementare precision/recall/F1.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- viene invocata dall'orchestrator tramite `scripts/run_extrinsic_eval.py`;
- consuma indici prodotti da `src/retrieval/`;
- legge query e qrels normalizzati generati dai processor in `src/datasets/`;
- salva CSV in `results/extrinsic/`.

Nota importante sullo stato attuale:
- l'evaluator `document_retrieval` ricarica l'embedder tramite `src.embeddings.factory.EmbedderFactory`, quindi supporta tutti gli embedder registrati nella factory;
- per i retriever sono supportati `numpy` e `faiss`;
- e' possibile campionare un sottoinsieme di query con `evaluation.extrinsic_tasks.document_retrieval.query_sample_size`.
