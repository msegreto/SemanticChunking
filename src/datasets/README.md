# Datasets Module

Questa cartella gestisce l'ingresso dei dataset nel framework: download, validazione dei file raw, caricamento in memoria e conversione in una struttura normalizzata riusabile dal resto del pipeline.

File presenti:
- `base.py`: definisce `BaseDatasetProcessor`, cioè il flusso comune per tutti i dataset. Qui sta la logica di reuse della cache normalizzata, salvataggio `documents.jsonl`, `queries.json`, `qrels/<split>.json` e `metadata.json`.
- `factory.py`: definisce `DatasetFactory`, il punto di ingresso usato dall'orchestrator per scegliere il processor corretto in base al nome del dataset.
- `beir.py`: implementazione per dataset nel formato `beir/<dataset_name>`.
- `msmarco.py`: processor per MSMARCO nel formato attualmente usato dal progetto.
- `download_utils.py`: utility condivise per download, estrazione archivi, manifest e validazione dei file attesi.
- `dummy.py`: processor placeholder, non adatto a un uso reale nello stato attuale.
- `backup.py`: copia più vecchia o parallela di parte della logica di download; oggi è ridondante rispetto a `download_utils.py`.
- `__init__.py`: abilita gli import del package.

Come si collega al resto del progetto:
- l'orchestrator chiama `DatasetFactory.create(...)` in base al YAML;
- gli output normalizzati vengono poi usati direttamente da `src/splitting/`;
- i dati persistiti finiscono soprattutto in `data/raw/` e `data/normalized/`;
- l'extrinsic evaluation legge spesso proprio i file normalizzati generati qui.

Questa cartella è quindi il ponte tra dataset esterni e formato interno del framework.
