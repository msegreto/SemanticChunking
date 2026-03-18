# Configs

Configurazioni YAML e cartelle dedicate alle future configurazioni per componenti del framework.

Le configurazioni operative sono in:
- `configs/experiments/` per i setup di pipeline;
- `configs/execution/` per i profili di esecuzione riusabili (referenziati da `execution_profile`);
- `configs/evaluations/` per i profili di valutazione riusabili (referenziati da `evaluation_profile` nei file esperimento).

Nota: il pipeline orchestrato è streaming-only; i YAML esperimento dovrebbero quindi includere i parametri di streaming (`execution.streaming_docs_per_run`, opzionalmente `execution.streaming_run_until_complete`).
