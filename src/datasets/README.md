# Datasets Module

This folder contains the dataset ingestion layer for the project.

Its purpose is to provide a unified interface for:
- checking whether a dataset is already available locally,
- downloading it if needed,
- loading it from its raw external format,
- converting it into a normalized internal representation,
- caching the normalized version for future runs.

The module is designed so that the rest of the pipeline does **not** need to know the original dataset format (TSV, JSONL, qrels layout, etc.). Dataset-specific logic is isolated inside dedicated processors, while the pipeline consumes a common normalized structure.

---

## Supported datasets

At the moment, the module supports:

- `msmarco-docs`
- `beir/<dataset_name>`

Examples:

- `msmarco-docs`
- `beir/scifact`
- `beir/hotpotqa`
- `beir/nfcorpus`

Dataset selection is handled by the factory:
- `msmarco-docs` is mapped explicitly,
- any dataset name starting with `beir/` is routed to the generic BEIR processor.  

This behavior is implemented in `factory.py`.

---

## Folder contents

### `base.py`
Defines the abstract base class `BaseDatasetProcessor`.

This class implements the common processing flow:

1. check whether a normalized version already exists,
2. load from normalized cache if available,
3. otherwise ensure the raw dataset is present,
4. load and normalize the raw dataset,
5. save the normalized cache for future runs.

It also provides shared utilities for:
- resolving raw and normalized directories,
- deciding whether to download missing data,
- deciding whether to reuse normalized cache,
- saving/loading normalized files.

### `factory.py`
Defines `DatasetFactory`, the entry point used by the pipeline to instantiate the correct dataset processor from the dataset name.

### `msmarco.py`
Contains `MSMarcoDatasetProcessor`, responsible for:
- validating the required MS MARCO files for the requested split,
- downloading the corpus and associated query/qrels files if missing,
- loading the raw TSV files,
- returning the normalized in-memory representation.

### `beir.py`
Contains `BEIRDatasetProcessor`, a single processor used for all BEIR datasets in standard BEIR format.

It:
- validates the expected raw files for the requested split,
- downloads the dataset archive if needed,
- extracts and loads `corpus.jsonl`, `queries.jsonl`, and `qrels/<split>.tsv`,
- returns the normalized representation.

### `download_utils.py`
Contains shared utilities for:
- downloading files,
- extracting archives,
- writing dataset manifests,
- validating raw dataset completeness,
- building dataset-specific download logic for MS MARCO and BEIR.

### `dummy.py`
Contains a placeholder processor used as a fallback or temporary stub during development.

---

## Processing flow

Every dataset processor follows the same high-level flow:

```text
dataset name
   ↓
DatasetFactory
   ↓
dataset processor
   ↓
[normalized cache exists?]
   ├── yes → load normalized
   └── no
         ↓
   ensure raw availability
         ↓
   load raw dataset
         ↓
   normalize to internal format
         ↓
   save normalized cache
         ↓
   return normalized in-memory object