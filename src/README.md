# Src

Questa è la cartella principale del codice applicativo del framework.

L'organizzazione segue le fasi del pipeline sperimentale: acquisizione dataset, splitting, routing, chunking, embedding, retrieval e evaluation. Il file che coordina tutto è `src/pipelines/experiment_orchestrator.py` (oggi in modalità streaming-only).

Sottocartelle principali:
- `datasets/`: download, validazione, caricamento raw e normalizzazione.
- `splitting/`: trasformazione dei documenti in unità testuali.
- `routing/`: eventuale instradamento delle unità prima del chunking.
- `chunking/`: costruzione dei chunk finali.
- `embeddings/`: encoding dei chunk in vettori.
- `retrieval/`: costruzione e interrogazione degli indici.
- `evaluation/`: metriche intrinsic ed extrinsic.
- `pipelines/`: orchestration end-to-end.

Questa cartella è il centro del progetto: gli script in `scripts/` la invocano, i file YAML in `configs/experiments/` la configurano, mentre `data/` e `results/` contengono gli artefatti persistiti delle esecuzioni.
