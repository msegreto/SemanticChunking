# Splitting Module

This module implements the **text splitting stage** of the pipeline.  
Its purpose is to transform normalized documents into smaller textual units that can later be used by the chunking and embedding stages.

All splitters implement the common interface defined in `BaseSplitter`.

---

## Architecture

The module is structured around three main components:

- **BaseSplitter** – abstract interface for all splitting strategies.
- **SentenceSplitter** – sentence-level segmentation of documents. it uses `en_core_web_sm` by `spacy`
- **PropositionSplitter** – placeholder for proposition-level splitting.
- **SplitterFactory** – factory used by the pipeline to instantiate the correct splitter.

The factory pattern allows selecting the splitting strategy dynamically from the configuration file.

---

## Base Interface

All splitters must implement:

```python
split(dataset_output, config) -> dict