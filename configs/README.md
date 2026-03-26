# Configs

Configurazioni YAML e cartelle dedicate alle future configurazioni per componenti del framework.

Le configurazioni operative sono in:
- `configs/experiments/` per i setup di pipeline;
- `configs/execution/` per i profili di esecuzione riusabili (referenziati da `execution_profile`);
- `configs/evaluations/` per i profili di valutazione riusabili (referenziati da `evaluation_profile` nei file esperimento).

Il pipeline orchestrato e' ora lineare:
- `dataset >> split >> router opzionale >> chunking >> intrinsic >> retrieval >> extrinsic`

La sezione `execution` controlla solo riuso e rebuild degli artefatti.
