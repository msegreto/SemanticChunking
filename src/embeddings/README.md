# Embeddings Module

Questa cartella genera embedding testuali a partire da chunk, frasi o altre unità con campo `text`.

File presenti:
- `base.py`: definisce `BaseEmbedder`, `EmbeddingItem`, `EmbeddingResult` e la funzione `normalize_items(...)`, che converte vari formati di input in un formato uniforme.
- `factory.py`: `EmbedderFactory`, usata dall'orchestrator per selezionare il modello dal YAML.
- `mpnet.py`: embedder basato su `sentence-transformers/all-mpnet-base-v2`.
- `bge.py`: embedder basato su `BAAI/bge-large-en-v1.5`.
- `stella.py`: embedder basato su `dunzhang/stella_en_1.5B_v5`.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- input principale: chunk prodotti da `src/chunking/`;
- output: embeddings e metadata consumati da `src/retrieval/` per costruire l'indice;
- riuso indiretto: l'extrinsic evaluation richiama di nuovo l'embedder per codificare le query.

Questa cartella è il punto in cui i chunk testuali diventano vettori numerici utilizzabili dal retrieval.
