# Thesis Repo

Framework modulare per esperimenti di chunking, embedding, retrieval e valutazione su dataset BEIR e MSMARCO.

Il flusso principale parte da `scripts/run_pipeline.py`, legge una configurazione in `configs/experiments/` e orchestra i moduli sotto `src/`.

Cartelle principali:
- `configs/`: configurazioni degli esperimenti.
- `src/`: codice applicativo del framework.
- `data/`: dataset raw, cache normalizzate e artefatti intermedi.
- `results/`: output delle valutazioni.
- `scripts/`: entrypoint CLI.
- `docs/`: note di progettazione.
- `tests/`: struttura test, oggi quasi vuota.
