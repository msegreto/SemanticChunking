# Retrieval Module

Questa cartella gestisce il retrieval batch dense FAISS come stage di pipeline. L'unica API pubblica e' [`RetrievalTransformer`](/Users/mattiasegreto/Desktop/TesiCode/src/retrieval/transformer.py).

Struttura:
- `transformer.py`: entrypoint pubblico. Esegue retrieval dense FAISS su chunk e query embeddizzati e, se disponibili i topics, produce anche una `run.tsv`.
- `__init__.py`: esporta `RetrievalTransformer`.

Contratto:
- input: `chunk_output` prodotto da `src/chunking/`
- opzionalmente anche un dataset PyTerrier-compatible per ottenere `topics` e `qrels`
- output: artefatti sotto `data/pt_indexes/...`, `items.jsonl`, `manifest.json` e, se presenti i topics, una `run.tsv`

Il backend disponibile e' `dense_faiss`: retrieval vettoriale con embedding di chunk e query, indicizzazione FAISS e distanza `cosine` o `l2`.
