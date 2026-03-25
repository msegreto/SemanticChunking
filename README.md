# Thesis Repo

Framework modulare per esperimenti di chunking, embedding, retrieval e valutazione su dataset BEIR e MSMARCO.

Il flusso principale parte da `scripts/run_pipeline.py`, legge una configurazione in `configs/experiments/` e orchestra i moduli sotto `src/`.
L'orchestrator attuale esegue il pipeline in modalità streaming incrementale (window-based).

## Nota compatibilita' Stella

Per l'embedder `stella` (`dunzhang/stella_en_1.5B_v5`) e' stata osservata un'incompatibilita' con versioni recenti di `transformers` (API cache Qwen2).  
Riferimento: https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47

Per questo motivo il progetto pinna `transformers` a una versione compatibile in `requirements.txt`.

Cartelle principali:
- `configs/`: configurazioni degli esperimenti.
- `src/`: codice applicativo del framework.
- `data/`: dataset raw, cache normalizzate e artefatti intermedi.
- `results/`: output delle valutazioni.
- `scripts/`: entrypoint CLI.
- `docs/`: note di progettazione.
- `tests/`: struttura test, oggi quasi vuota.
