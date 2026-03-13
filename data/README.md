# Data

Questa cartella contiene tutti gli artefatti persistiti del pipeline.

È il punto in cui il framework conserva:
- dataset raw scaricati o preparati;
- cache normalizzate;
- output intermedi dello splitting;
- chunk salvati su disco;
- eventuali embeddings;
- indici di retrieval.

Sottocartelle principali:
- `raw/`: sorgenti nel formato originale o quasi-originale;
- `normalized/`: formato interno standard usato dal framework;
- `processed/`: output dello splitting;
- `chunks/`: chunk salvati su disco;
- `indexes/`: indici retrieval;
- `embeddings/`: area prevista per embeddings persistiti;
- `splits/`: cartella placeholder non usata dal flow attuale.

Questa cartella è alimentata soprattutto da `src/datasets/`, `src/splitting/`, `src/chunking/` e `src/retrieval/`.
