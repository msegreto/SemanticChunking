# Splitting Module

Questa cartella implementa la fase di splitting, cioè la trasformazione dei documenti normalizzati in unità testuali più piccole che possono poi essere chunkate.

File presenti:
- `base.py`: interfaccia astratta `BaseSplitter` (API streaming).
- `factory.py`: `SplitterFactory`, usata dall'orchestrator per selezionare lo splitter dal YAML.
- `sentence.py`: implementazione principale nel flusso corrente. Espone i metodi incrementali `build_streaming_components(...)` e `split_document_streaming(...)`.
- `proposition.py`: placeholder non supportato nel flusso streaming (al momento solleva errore esplicito).
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: dati normalizzati prodotti da `src/datasets/`;
- output: struttura consumata da `src/routing/` o direttamente da `src/chunking/`;
- persistenza: quando `split.save_output` è abilitato, il pipeline salva in `data/processed/`.

Nel flusso attuale il file davvero centrale è `sentence.py` con `type: sentence`.
