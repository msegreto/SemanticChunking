# Retrieval Module

Questa cartella gestisce costruzione, salvataggio, caricamento e interrogazione degli indici vettoriali.

File presenti:
- `base.py`: interfaccia astratta `BaseRetriever`.
- `factory.py`: `RetrieverFactory`, usata dall'orchestrator per selezionare backend `numpy` o `faiss`.
- `numpy_retriever.py`: implementazione semplice basata su array NumPy; salva `vectors.npy`, `metadata.pkl` e `manifest.json`.
- `faiss_retriever.py`: implementazione basata su FAISS per indicizzazione e ricerca più specializzata.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: embeddings prodotti da `src/embeddings/`;
- output persistito: artefatti sotto `data/indexes/...`;
- uso successivo: `scripts/run_extrinsic_eval.py` e `src/evaluation/extrinsic/` caricano questi indici per misurare la retrieval quality.

Nel baseline osservato il backend usato è `numpy`.
