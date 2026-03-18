# Datasets Module

Questa cartella gestisce il primo blocco reale del pipeline: acquisizione del dataset, verifica della sua consistenza, caricamento in memoria e conversione in un formato normalizzato unico per il resto del framework.

In pratica, tutto quello che viene dopo, cioè:
- `src/splitting/`
- `src/chunking/`
- `src/embeddings/`
- `src/retrieval/`
- `src/evaluation/extrinsic/`

si aspetta di ricevere dati prodotti da questo modulo con una struttura coerente.

## Ruolo del modulo

Il modulo dataset ha quattro responsabilità principali:

1. capire quale dataset è stato richiesto dal file YAML;
2. verificare se esiste già una cache `normalized` riusabile;
3. se la cache non è disponibile o non è compatibile, caricare la versione `raw` o scaricarla;
4. restituire sempre un payload uniforme, indipendentemente dalla sorgente originaria.

Questo significa che il resto del progetto non deve conoscere il formato originale dei file esterni, né sapere se i dati arrivano da BEIR, da MS MARCO reale o da una cache locale già costruita.

## File presenti

### `base.py`
È il file più importante del modulo.

Definisce `BaseDatasetProcessor`, cioè il comportamento comune di tutti i processor dataset.

Responsabilità principali:
- risolvere le cartelle `raw` e `normalized`;
- decidere se riusare la cache normalizzata;
- controllare se la cache è completa e compatibile;
- caricare i dati da `normalized`;
- altrimenti delegare al processor concreto il caricamento della `raw`;
- validare il payload finale;
- salvare il formato normalizzato;
- opzionalmente supportare la rimozione della `raw` dopo la creazione corretta della `normalized`.

Il formato normalizzato atteso è:

- `documents.jsonl`
- `queries.json`
- `qrels/<split>.json`
- `metadata.json`

Il payload restituito in memoria deve sempre contenere:
- `documents`
- `queries`
- `qrels`
- `metadata`

### `factory.py`
Definisce `DatasetFactory`.

È il punto di ingresso usato da [src/pipelines/experiment_orchestrator.py](/Users/mattiasegreto/Desktop/TesiCode/src/pipelines/experiment_orchestrator.py).

La factory ha il compito di scegliere il processor corretto in base al nome del dataset richiesto nel YAML.

Attualmente distingue in modo esplicito:
- `beir/<dataset_name>` -> `BEIRDatasetProcessor`
- `msmarco` -> `MSMarcoDatasetProcessor`
- `msmarco-docs` -> `MSMarcoDatasetProcessor`

Scelta importante del progetto:
- `beir/msmarco` viene trattato come un normale dataset BEIR;
- `msmarco` e `msmarco-docs` rappresentano il dataset MS MARCO reale.

Questa distinzione è fondamentale per evitare ambiguità tra dataset con nomi simili ma sorgenti e formati diversi.

### `beir.py`
Contiene `BEIRDatasetProcessor`.

È il processor generico per tutti i dataset in formato `beir/<dataset_name>`.

Responsabilità:
- verificare che la raw BEIR sia completa;
- scaricare il dataset dal mirror BEIR se manca;
- leggere:
  - `corpus.jsonl`
  - `queries.jsonl`
  - `qrels/<split>.tsv`
- convertire il tutto nel formato normalizzato interno.

È un processor volutamente generico: il dataset specifico viene passato come parametro del costruttore e non richiede una classe dedicata per ogni collezione BEIR.

### `scripted.py`
Contiene processor specializzati che costruiscono direttamente la cache normalizzata tramite script dedicati.

Attualmente usato per:
- `beir/qasper` -> `scripts/download_qasper_hf_to_normalized.py`
- `beir/techqa` -> `scripts/download_techqa_hf_to_normalized.py`

Comportamento:
- se la normalized richiesta esiste ed è compatibile, viene riusata;
- altrimenti il processor lancia lo script di conversione e poi carica la normalized appena creata;
- questo permette run end-to-end da YAML senza pre-download manuale.

### `msmarco.py`
Contiene `MSMarcoDatasetProcessor`.

Questo processor oggi è dedicato al MS MARCO reale, non alla variante BEIR.

Responsabilità:
- verificare la presenza dei file raw ufficiali di MS MARCO;
- scaricarli tramite `download_msmarco_documents(...)` se necessario;
- leggere:
  - `msmarco-docs.tsv`
  - query TSV dello split richiesto
  - qrels TSV dello split richiesto, se presenti
- produrre un payload coerente con il formato interno del framework.

Aspetto importante:
- il processor accetta sia `msmarco` sia `msmarco-docs` come nomi richiesti;
- nei metadata salva sia il nome richiesto sia un nome canonico, per rendere più chiaro il significato della cache.

### `download_utils.py`
Contiene la logica condivisa di supporto per i dataset.

Responsabilità:
- creare directory;
- scaricare file remoti;
- estrarre ZIP e GZIP;
- scrivere `manifest.json`;
- verificare la completezza delle strutture raw;
- definire i file richiesti per BEIR e MS MARCO;
- implementare il download BEIR e il download MS MARCO.

Questo file è la fonte unica di verità per la logica di download del modulo.

### `__init__.py`
File standard di package.

Non contiene logica applicativa, ma rende importabile il modulo `src.datasets`.

## Flusso logico attuale

Il comportamento desiderato del modulo è il seguente:

1. ricevere il config dataset dal file experiment YAML;
2. risolvere `split`, `raw_dir` e `normalized_dir`;
3. verificare se esiste una versione `normalized`;
4. se esiste, verificare che sia compatibile con:
   - dataset richiesto
   - split richiesto
   - schema version
   - struttura dei file normalizzati
5. se la `normalized` è compatibile, usarla e fermarsi lì;
6. altrimenti verificare o preparare la `raw`;
7. caricare la `raw`;
8. validare il payload;
9. salvare la `normalized`;
10. opzionalmente eliminare la `raw` se la politica del progetto lo prevede.

La regola di progetto più importante è questa:

- una volta creata correttamente la `normalized`, questa può diventare la fonte primaria e sufficiente;
- la `raw` può essere considerata transitoria;
- nelle esecuzioni successive il controllo dovrebbe fermarsi sulla `normalized`, salvo rebuild forzato o incompatibilità.
- nel comportamento attuale del modulo, la `raw` viene eliminata di default dopo la creazione corretta della `normalized`, salvo override esplicito via config.

## Struttura del payload normalizzato

Tutti i processor devono restituire una struttura coerente con questa forma:

```python
{
    "documents": {
        "<doc_id>": {
            ...
        }
    },
    "queries": {
        "<query_id>": "<query_text>"
    },
    "qrels": {
        "<query_id>": {
            "<doc_id>": <relevance_int>
        }
    },
    "metadata": {
        ...
    }
}
```

Il `BaseDatasetProcessor` oggi valida esplicitamente:
- presenza delle chiavi principali;
- tipo di `documents`, `queries`, `qrels`, `metadata`;
- tipo degli identificativi;
- tipo dei testi query;
- tipo dei valori di relevance nei qrels.

Questo riduce la probabilità che errori di parsing dataset si propaghino alle fasi successive.

## Metadata e cache

I metadata salvati in `metadata.json` hanno un ruolo importante:
- descrivono il dataset;
- documentano da dove arrivano i dati;
- permettono di decidere se una cache `normalized` è riusabile;
- aiutano il debugging senza dover riaprire i file raw.

I campi più importanti sono:
- `dataset_name`
- `split`
- `processor`
- `raw_path`
- `normalized_path`
- `normalized_schema_version`
- `normalized_files`

Nel caso di MSMARCO reale può comparire anche:
- `canonical_dataset_name`

## Collegamenti con il resto del progetto

Questo modulo è collegato direttamente a:

- [scripts/run_pipeline.py](/Users/mattiasegreto/Desktop/TesiCode/scripts/run_pipeline.py)
  perché da lì parte l'orchestrator;
- [scripts/download_datasets.py](/Users/mattiasegreto/Desktop/TesiCode/scripts/download_datasets.py)
  che usa la factory per preparare le raw;
- [src/pipelines/experiment_orchestrator.py](/Users/mattiasegreto/Desktop/TesiCode/src/pipelines/experiment_orchestrator.py)
  che chiama `DatasetFactory.create(...)` e poi `processor.process(...)`;
- [src/splitting/](/Users/mattiasegreto/Desktop/TesiCode/src/splitting/README.md)
  che usa il payload dataset come input;
- [src/evaluation/extrinsic/io.py](/Users/mattiasegreto/Desktop/TesiCode/src/evaluation/extrinsic/io.py)
  che legge query e qrels normalizzati dal disco.

Dal punto di vista dei dati persistiti, le cartelle coinvolte sono soprattutto:
- [data/raw](/Users/mattiasegreto/Desktop/TesiCode/data/raw/README.md)
- [data/normalized](/Users/mattiasegreto/Desktop/TesiCode/data/normalized/README.md)

## Scelte progettuali attuali

Le scelte attuali del modulo dataset sono:

- meglio fallire con errore esplicito che usare un fallback fittizio;
- meglio avere un formato normalizzato unico che tanti parser sparsi nel resto del codice;
- meglio distinguere chiaramente MS MARCO reale e `beir/msmarco`;
- meglio trattare la `normalized` come fonte primaria una volta costruita correttamente.

Queste scelte vanno nella direzione di rendere il framework più robusto e meno ambiguo nelle fasi successive della tesi.
