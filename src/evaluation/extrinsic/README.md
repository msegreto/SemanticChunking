# Extrinsic Evaluation

Questa cartella contiene la valutazione extrinsic come stage di pipeline. L'unica API pubblica e' [`ExtrinsicEvaluationTransformer`](/Users/mattiasegreto/Desktop/TesiCode/src/evaluation/extrinsic/transformer.py).

Struttura:
- `transformer.py`: entrypoint pubblico. Coordina i task extrinsic richiesti e salva il CSV finale.
- `document_retrieval.py`: implementazione interna del task document retrieval su `run.tsv` PyTerrier.
- `evidence_retrieval.py`: implementazione interna del task evidence retrieval su `run.tsv` PyTerrier.
- `answer_generation.py`: implementazione interna del task answer generation su `run.tsv` PyTerrier.
- `common.py`: helper condivisi per contesto di run e costruzione righe output.
- `io.py`: risolve i path a query/qrels/annotations e carica `run.tsv` e `items.jsonl`.
- `__init__.py`: export del package.

Collegamenti:
- viene invocata dall'orchestrator tramite `ExtrinsicEvaluationTransformer`;
- consuma l'output del retrieval PyTerrier (`manifest.json`, `run.tsv`, `items.jsonl`);
- legge query e qrels normalizzati generati in `src/datasets/`;
- salva CSV in `results/extrinsic/`.

Nota importante:
- i task extrinsic attivi sono selezionati da `evaluation.extrinsic_tasks_to_run` (lista). Se assente, fallback a `evaluation.extrinsic_evaluator`.
- per `answer_generation`, il modello BERTScore paper-aligned e' `microsoft/deberta-xlarge-mnli`;
- se BERTScore non e' disponibile, la run viene marcata `partial`.
- `answer_generation` usa `pyterrier_rag` come unica implementazione supportata.
- campi utili in `evaluation.extrinsic_tasks.answer_generation`: `reader_backend` (oggi supportato solo `seq2seqlm`), `reader_model` e `top_k`.
- `evaluation.extrinsic_tasks.answer_generation.top_k` controlla quanti chunk recuperati usare per generare la risposta (default `5`).

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
