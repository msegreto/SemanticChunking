# Configs

Configurazioni YAML e cartelle dedicate alle future configurazioni per componenti del framework.

Le configurazioni operative sono in `configs/experiments/`.

Nota: il pipeline orchestrato è streaming-only; i YAML esperimento dovrebbero quindi includere i parametri di streaming (`execution.streaming_docs_per_run`, opzionalmente `execution.streaming_run_until_complete`).
