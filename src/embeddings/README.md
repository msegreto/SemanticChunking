# Embeddings Module

Questa cartella genera embedding testuali a partire da chunk, frasi o altre unità con campo `text`.

File presenti:
- `base.py`: definisce `BaseEmbedder`, `EmbeddingItem` e le funzioni comuni di validazione/normalizzazione (`normalize_items(...)`, `validate_embedder_config(...)`, `validate_embedding_output(...)`).
- `sentence_transformer_base.py`: base class condivisa per gli embedder basati su `SentenceTransformer`; espone anche un livello piu' basso riusabile via `encode_texts(...)`.
- `factory.py`: `EmbedderFactory`, usata dall'orchestrator e dai chunker semantici per selezionare l'embedder dal YAML.
- `mpnet.py`: embedder logico `mpnet`, che incapsula il model id Hugging Face effettivo.
- `bge.py`: embedder basato su `BAAI/bge-large-en-v1.5`.
- `stella.py`: embedder basato su `dunzhang/stella_en_1.5B_v5`.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- input principale: chunk prodotti da `src/chunking/`;
- output: un `dict[str, Any]` validato con `embeddings`, `items` e `metadata`, consumato da `src/retrieval/` per costruire l'indice;
- riuso indiretto: l'extrinsic evaluation richiama di nuovo l'embedder per codificare le query.

Nota architetturale:

gli embedder possono ora essere riusati anche a un livello piu' basso tramite `encode_texts(...)` / `encode_items(...)` e l'adapter `as_langchain_embeddings(...)`. Questo serve soprattutto al semantic chunking, dove bisogna embeddare sentence o split units prima che esistano i chunk finali.

Questa cartella è il punto in cui i chunk testuali diventano vettori numerici utilizzabili dal retrieval.
