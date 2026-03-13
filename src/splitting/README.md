# Splitting Module

Questa cartella implementa la fase di splitting, cioè la trasformazione dei documenti normalizzati in unità testuali più piccole che possono poi essere chunkate.

File presenti:
- `base.py`: interfaccia astratta `BaseSplitter`.
- `factory.py`: `SplitterFactory`, usata dall'orchestrator per selezionare lo splitter dal YAML.
- `sentence.py`: implementazione principale. Legge i documenti normalizzati, usa spaCy o un sentencizer minimale e produce `split_units`, statistiche per documento e un file JSONL opzionale in `data/processed/...`.
- `proposition.py`: placeholder per uno split a livello di proposizione. Oggi restituisce output vuoto.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: dati normalizzati prodotti da `src/datasets/`;
- output: struttura consumata da `src/routing/` o direttamente da `src/chunking/`;
- persistenza: quando `save_output` è abilitato, salva in `data/processed/`.

Nel flusso attuale il file davvero centrale è `sentence.py`, perché `configs/experiments/base.yaml` usa `type: sentence`.
