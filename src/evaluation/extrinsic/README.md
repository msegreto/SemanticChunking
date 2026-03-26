# Extrinsic Evaluation

Questa cartella contiene la valutazione extrinsic, cioè la misurazione del comportamento dei chunk su un task downstream reale.

Nel progetto attuale il task implementato è soprattutto il document retrieval.

File presenti:
- `base.py`: interfaccia `BaseExtrinsicEvaluator`.
- `common.py`: helper condivisi per contesto di run e costruzione righe output.
- `factory.py`: factory per scegliere l'evaluator richiesto.
- `document_retrieval.py`: evaluator principale. Carica query e qrels, ricarica embedder e retriever, esegue la ricerca e calcola macro+micro precision/recall/F1 e DCG/NDCG a vari `k` (NDCG via `ir_measures`).
- `evidence_retrieval.py`: evaluator evidence-level. Recupera top-k chunk e calcola precision/recall/F1 sentence-level misurando quante evidence sentence gold sono presenti nei chunk recuperati.
- `answer_generation.py`: evaluator answer-level. Genera risposta dai top-5 chunk (paper setting) e calcola `qa_similarity` (coseno query-risposta) + `bertscore_f1`.
- `io.py`: risolve i path a query, qrels e metadata indice, e carica questi file dal disco.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- viene invocata dall'orchestrator tramite `scripts/run_extrinsic_eval.py`;
- consuma indici prodotti da `src/retrieval/`;
- legge query e qrels normalizzati generati dai processor in `src/datasets/`;
- salva CSV in `results/extrinsic/`.

Nota importante sullo stato attuale:
- l'evaluator `document_retrieval` ricarica l'embedder tramite `src.embeddings.factory.EmbedderFactory`, quindi supporta tutti gli embedder registrati nella factory;
- per i retriever sono supportati `numpy` e `faiss`;
- i task extrinsic attivi sono selezionati da `evaluation.extrinsic_tasks_to_run` (lista). Se assente, fallback retrocompatibile a `evaluation.extrinsic_evaluator`.
- per `answer_generation`, il modello BERTScore paper-aligned e' `microsoft/deberta-xlarge-mnli`;
- se `generation_model.name` non e' `gpt-4o-mini`, la run viene marcata `partial` per segnalare che la generazione non e' identica al paper.

Scelta di aggregazione chunk -> documento (per metriche doc-level come DCG/NDCG):
- il retrieval opera sui chunk, mentre la valutazione `document_retrieval` e' a livello documento;
- quando piu' chunk appartengono allo stesso documento, il punteggio documento viene aggregato con strategia `max`;
- quindi, per ogni query, il punteggio del documento e' il massimo score tra i chunk recuperati di quel documento.

Opzioni utili per `document_retrieval` in `evaluation.extrinsic_tasks.document_retrieval`:
- `query_subset_size`: intero > 0. Se presente, valuta anche un sottoinsieme casuale di query (es. `100`).
- `query_subset_seed`: seed del sampling (default `42`).
- `include_full_query_set`: se `true` (default), oltre al subset calcola anche il full set.

Nel CSV di output, per ciascun `k`, vengono salvati:
- colonne legacy `precision`, `recall`, `f1` (allineate alle metriche macro);
- nuove colonne `macro_precision`, `macro_recall`, `macro_f1`;
- nuove colonne `micro_precision`, `micro_recall`, `micro_f1`;
- metadati subset: `query_subset`, `query_subset_size_requested`, `query_subset_seed`.
