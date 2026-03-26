# Datasets Module

Questa cartella gestisce il confine tra i dataset esterni e il formato interno usato dal pipeline PyTerrier-oriented del progetto.

L'API pubblica attuale e' piccola:
- `src.datasets.process_dataset(config)`
- `src.datasets.try_load_normalized(config)`
- `src.datasets.ensure_dataset_prepared(config)`
- `src.datasets.get_dataset(config)`
- `src.datasets.PyTerrierNormalizedDataset`

## Obiettivo del modulo

Il modulo ha tre responsabilita' principali:

1. risolvere il dataset richiesto dal config;
2. costruire o riusare una cache `normalized` coerente;
3. esporre quella cache sia come payload Python sia come dataset compatibile con PyTerrier.

Il resto del progetto non deve conoscere la sorgente originaria del dataset. Dopo questa fase tutto converge su una struttura normalizzata comune.

## File presenti

### `service.py`

E' l'entrypoint reale del modulo.

Contiene:
- `process_dataset(...)`: prepara o riusa il dataset normalizzato e restituisce il payload completo;
- `try_load_normalized(...)`: prova a caricare una cache gia' pronta se compatibile;
- `ensure_dataset_prepared(...)`: garantisce la presenza del dataset e restituisce il path della cache;
- `get_dataset(...)`: costruisce l'oggetto `PyTerrierNormalizedDataset`;
- la risoluzione delle specifiche dataset (`beir/...`, `msmarco`, casi script-based come `beir/qasper` e `beir/techqa`).

Il service oggi supporta due modalita' di costruzione:
- `kind="pyterrier"`: dataset ottenuti tramite `pt.get_dataset(...)`;
- `kind="script"`: dataset costruiti da converter dedicati, usati per i casi che non passano bene dal flusso PyTerrier puro.

### `pt_dataset.py`

Definisce `PyTerrierNormalizedDataset`, che incapsula una directory `normalized` e la rende usabile sia dal codice interno sia dalle fasi di pipeline che si aspettano metodi stile PyTerrier.

Metodi principali:
- `get_corpus_iter()`
- `get_topics()`
- `get_qrels()`
- `load_payload()`
- `exists()`
- `load_metadata()`

Il payload prodotto da `load_payload()` contiene:
- `corpus`
- `topics`
- `qrels`
- `metadata`
- `pt_dataset`
- opzionalmente `evidences`
- opzionalmente `answers`

### `__init__.py`

Riesporta l'API pubblica del modulo.

## Formato normalized attuale

Una cache dataset valida contiene almeno:
- `corpus.jsonl`
- `topics.jsonl`
- `qrels/<split>.tsv`
- `metadata.json`

Possono esistere anche:
- `evidences.json`
- `answers.json`

Lo schema version usato dal service e' `2.0`.

## Flusso operativo

Quando l'orchestrator chiama `process_dataset(config)`:

1. il service risolve nome dataset, split e cartella normalized;
2. se la cache esiste e il metadata e' compatibile, la riusa;
3. altrimenti ricostruisce la cache:
   - via PyTerrier per i dataset supportati da `pt.get_dataset(...)`;
   - via script dedicato per dataset come `beir/qasper` e `beir/techqa`;
4. ricarica la cache normalizzata appena prodotta;
5. restituisce il payload Python con `pt_dataset` incluso.

Questo rende uniforme il passaggio alla fase successiva di `split`.

## Campi config effettivamente rilevanti

Nel contratto attuale i campi dataset davvero usati sono:
- `name`
- `split`
- `normalized_dir`
- `download_if_missing`
- i parametri opzionali specifici dei converter script-based, quando applicabili

Campi legacy come `data_dir`, `use_normalized_if_available` e `save_normalized` non fanno piu' parte del comportamento runtime e non dovrebbero essere reintrodotti nei nuovi YAML.

## Collegamento con il resto del progetto

Il modulo dataset alimenta direttamente:
- [`src/pipelines/experiment_orchestrator.py`](/Users/mattiasegreto/Desktop/TesiCode/src/pipelines/experiment_orchestrator.py)
- [`src/splitting/transformer.py`](/Users/mattiasegreto/Desktop/TesiCode/src/splitting/transformer.py)
- [`src/evaluation/extrinsic/io.py`](/Users/mattiasegreto/Desktop/TesiCode/src/evaluation/extrinsic/io.py)

Il punto chiave e' questo:
- lo split puo' iterare sul corpus tramite `pt_dataset.get_corpus_iter()`;
- il retrieval puo' ottenere topics e qrels tramite `pt_dataset.get_topics()` e `pt_dataset.get_qrels()`;
- le valutazioni extrinsic possono leggere i file normalizzati dal disco quando serve.

## Note pratiche

- `beir/<dataset_name>` viene di norma risolto via PyTerrier come `irds:beir/<dataset_name>`.
- `msmarco` e `msmarco-docs` vengono risolti come `irds:msmarco-document`.
- `beir/qasper` e `beir/techqa` restano casi speciali gestiti da script di conversione dedicati.

In sintesi: questo modulo non implementa piu' una gerarchia di processor/factory legacy; oggi e' un service sottile che produce una cache normalized e un wrapper `PyTerrierNormalizedDataset` su cui si appoggia il resto del pipeline.
