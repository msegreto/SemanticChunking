# Chunking Module

Questa cartella implementa le strategie di chunking, cioè la trasformazione delle unità prodotte dallo splitting in chunk finali da usare per embedding, retrieval e valutazione.

File presenti:
- `base.py`: interfaccia astratta `BaseChunker`.
- `factory.py`: `ChunkerFactory`, usata dall'orchestrator per selezionare la strategia richiesta nel YAML.
- `fixed.py`: implementazione concreta oggi realmente funzionante. Raggruppa le sentence per documento, le divide in un numero prefissato di chunk e può salvarle su disco in `data/chunks/`.
- `semantic.py`: placeholder per una strategia semantica futura. Attualmente restituisce `chunks` vuoto.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- input: `split_units` prodotti da `src/splitting/` o passati attraverso `src/routing/`;
- output: lista di chunk usata da `src/evaluation/intrinsic/`, `src/embeddings/` e indirettamente da `src/retrieval/`;
- persistenza: `fixed.py` salva artefatti in `data/chunks/<dataset>_fixed/`.

Nel run baseline attuale, questa cartella è cruciale perché `fixed.py` genera i chunk su cui si basano tutte le fasi successive.
