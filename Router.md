# Router Module

## Overview

This module implements the **MoC-style router** for the chunking framework.

The router is **not** responsible for generating chunk boundaries directly. Its task is to analyze an input document and predict the **most suitable chunk granularity class**, then dispatch the document to the corresponding chunking expert.

This design follows the core idea of the MoC paper: a **multi-granularity-aware router** selects the best expert, while the actual chunk generation is delegated to specialized chunkers. The router therefore acts as a **document-level classifier**, not as a chunk boundary detector.

---

## Objectives

The router module should satisfy the following goals:

1. **Predict chunk granularity**
   - classify a document into one of a fixed number of granularity buckets

2. **Route documents to experts**
   - map the predicted label to a chunking expert

3. **Remain modular**
   - the router must be independent from the internal implementation of the experts

4. **Separate training from inference**
   - router training code must not be mixed with runtime pipeline logic

5. **Be reproducible**
   - all preprocessing, labels, configs, and artifacts must be saved and versioned

6. **Be robust at runtime**
   - the framework must still work even if the router fails or is unavailable

---

## What the paper tells us

The paper provides enough information for a **faithful reimplementation**, but not for an exact source-level reproduction.

### What is clearly specified

The router is trained as a **4-class classifier** over chunk granularity labels.

The labels are assigned using the **average chunk length** of GPT-4o-segmented texts. The four classes correspond to these intervals:

- label `0` -> `(0, 120]`
- label `1` -> `(120, 150]`
- label `2` -> `(150, 180]`
- label `3` -> `(180, +∞)`

The training inputs are normalized so that their lengths stay around **1024 characters**, by truncating long texts and concatenating short ones while keeping text chunks as the operational unit.

The training loss is standard **multiclass cross-entropy**.

The appendix also provides a first baseline for training:

- batch size per device = `3`
- gradient accumulation = `16`
- learning rate = `1e-5`
- epochs = `3`
- scheduler = `cosine`
- warmup ratio = `0.1`
- bf16 enabled

### What is missing

The paper does **not** clearly specify:

- the exact backbone SLM used for the router
- the exact tokenizer settings
- the exact classification head
- the exact artifact format
- the exact data serialization format

Because of this, the implementation should be documented as:

> **paper-aligned router reimplementation**

and not as an exact reproduction of the original source code.

---

## Role of the router inside the framework

The router sits **between document splitting and chunking**.

A simplified view is:

```text
Document
  -> splitter / preprocessor
  -> router
  -> selected expert
  -> chunk generation
  -> post-processing
  -> downstream evaluation
````

This means the router only decides:

> “Which chunking expert should process this document?”

It does **not** decide:

* where a chunk starts
* where a chunk ends
* how many chunks to create exactly

Those decisions belong to the selected expert.

---

## Proposed project structure

A clean project structure should separate router code into:

* shared abstractions
* training
* inference
* model implementations
* utilities
* configs
* scripts
* artifacts
* data

Recommended structure:

```text
src/
  router/
    README.md
    __init__.py

    base.py
    types.py
    registry.py
    constants.py

    inference/
      __init__.py
      router.py
      loader.py
      predictor.py
      preprocessing.py
      fallback.py
      mapping.py

    training/
      __init__.py
      dataset_builder.py
      labeler.py
      normalizer.py
      collator.py
      trainer.py
      evaluator.py
      split.py
      sampler.py
      diagnostics.py

    models/
      __init__.py
      hf_sequence_classifier.py

    utils/
      __init__.py
      io.py
      logging.py
      seeds.py
      validation.py

configs/
  router/
    train.yaml
    infer.yaml

scripts/
  train_router.py
  eval_router.py
  inspect_router_data.py

data/
  router/
    raw/
    processed/
    splits/
    manifests/

artifacts/
  router/
    <router_name>/
      config.json
      label_map.json
      preprocess.json
      metrics.json
      tokenizer/
      checkpoints/
      best/
```

---

## Why this structure

This structure is recommended because it keeps responsibilities clearly separated.

### `src/router/base.py`

Defines the abstract router interface.

This should expose methods like:

* `predict_label(text)`
* `predict_proba(text)`
* `load(path)`
* `save(path)`

The base interface should remain independent from HuggingFace or any specific model library.

### `src/router/types.py`

Contains typed data structures for:

* router configuration
* predictions
* training examples
* processed dataset rows
* batch outputs

### `src/router/registry.py`

Maps config names to concrete implementations.

Example:

* `hf-seq-cls` -> `HFSequenceClassifierRouter`

This allows the orchestrator to instantiate routers by config.

### `src/router/inference/`

Contains all runtime code used during actual chunking.

This must be lightweight and must only:

* load the artifact
* preprocess the input
* run inference
* return the predicted expert

### `src/router/training/`

Contains all code needed for:

* data preparation
* label construction
* normalization
* fine-tuning
* evaluation
* diagnostics

Training code should not be imported by default in the production pipeline.

### `src/router/models/`

Contains concrete router implementations.

For the first version, one implementation is enough:

* a HuggingFace sequence classifier with 4 output classes

### `scripts/`

Must contain standalone entry points for:

* training
* intrinsic evaluation
* dataset inspection

### `artifacts/`

Stores trained models and all metadata needed for exact reuse.

---

## Data requirements

The router is trained from **reference chunked data**.

This means the router does not need manually annotated “best chunk size” labels. Instead, it learns from documents that already have a trusted chunk decomposition.

### Minimum raw data per example

Each training source example should contain at least:

```json
{
  "doc_id": "doc_001",
  "raw_text": "full original text...",
  "reference_chunks": [
    "chunk one...",
    "chunk two...",
    "chunk three..."
  ],
  "source_dataset": "crud"
}
```

### Why this is enough

From this structure, we can derive:

* average chunk length
* granularity label
* normalized router input text
* chunk count
* source dataset metadata

That is all we need to create the router training set.

---

## Label generation

The paper derives labels from the **average chunk length** of GPT-4o-generated chunked text.

The labeling rule should therefore be isolated in a dedicated file and implemented exactly once.

### Label assignment rule

```text
if 0 < avg_chunk_length <= 120:
    label = 0
elif 120 < avg_chunk_length <= 150:
    label = 1
elif 150 < avg_chunk_length <= 180:
    label = 2
else:
    label = 3
```

### Why isolate this logic

This is important because later you may want to compare:

* the original paper bins
* quantile-based bins
* dataset-specific bins
* a different number of classes

So the labeling logic should not be buried inside the trainer.

---

## Input normalization

A crucial part of the MoC router is the input normalization step.

The paper states that long texts are **truncated** and short texts are **concatenated**, keeping text chunks as the operational unit, so that the resulting training examples stay around **1024 characters**.

### Why this matters

Without normalization, the router could learn a trivial shortcut:

* long document -> large chunk label
* short document -> small chunk label

This would mean the model is learning document length rather than real granularity-related features.

The purpose of normalization is therefore to make the router focus on:

* semantic cues
* discourse structure
* textual patterns correlated with chunk granularity

and not just raw input size.

### Where this logic should live

It should be implemented in:

```text
src/router/training/normalizer.py
```

and the exact normalization metadata should be saved into:

```text
artifacts/router/<router_name>/preprocess.json
```

---

## Recommended processed dataset schema

After preprocessing, each row should look like this:

```json
{
  "doc_id": "doc_001",
  "input_text": "normalized text around 1024 chars...",
  "label": 2,
  "avg_chunk_length": 164.7,
  "num_reference_chunks": 7,
  "source_dataset": "crud"
}
```

This schema is compact, debuggable, and sufficient for training.

---

## Data pipeline

The recommended dataset preparation pipeline is the following.

### Step 1 — collect raw routed examples

Input:

* original document text
* trusted reference chunks

### Step 2 — compute average chunk length

For each example:

```text
avg_chunk_length = mean(length(chunk) for chunk in reference_chunks)
```

### Step 3 — assign granularity label

Use the 4 paper-defined intervals.

### Step 4 — normalize router input

Generate `input_text` by:

* truncating long examples
* concatenating short examples
* preserving chunk units whenever possible

### Step 5 — save processed examples

Save the processed rows in:

* `jsonl` for simplicity
* `parquet` later if needed for scale

### Step 6 — create splits

Create:

* train
* validation
* test

The split must happen by **document identity**, not after random row mixing.

---

## Recommended data layout

```text
data/router/
  raw/
    crud_router_source.jsonl
    webcpm_router_source.jsonl

  processed/
    router_examples.jsonl

  splits/
    train.jsonl
    valid.jsonl
    test.jsonl

  manifests/
    label_distribution.json
    preprocessing_report.json
    source_stats.json
```

---

## Training design

The first router implementation should be intentionally simple.

### Recommended first model

A standard **HuggingFace sequence classification model** with:

* one encoder
* one classification head
* `num_labels = 4`

This is the most direct implementation of the paper’s formulation.

### Inputs and targets

Input:

* normalized text

Target:

* one integer label in `{0, 1, 2, 3}`

Loss:

* multiclass cross-entropy

### Initial training hyperparameters

The first baseline should use the paper settings:

```yaml
epochs: 3
learning_rate: 1e-5
train_batch_size: 3
eval_batch_size: 3
gradient_accumulation_steps: 16
scheduler: cosine
warmup_ratio: 0.1
bf16: true
```

These should be treated as a **paper-aligned baseline**, not necessarily as the final best settings for your framework.

---

## Training responsibilities by file

### `dataset_builder.py`

Builds the processed dataset from raw examples.

Responsibilities:

* read raw examples
* compute average chunk length
* assign labels
* normalize input text
* save processed dataset

### `labeler.py`

Contains only label assignment logic.

Responsibilities:

* define granularity bins
* assign class index
* validate interval consistency

### `normalizer.py`

Contains the router-specific normalization logic.

Responsibilities:

* truncate long examples
* concatenate short examples
* preserve chunk units when possible
* record stats

### `collator.py`

Batch collation for tokenized training batches.

Responsibilities:

* tokenize
* pad
* construct attention masks
* output tensors

### `trainer.py`

Runs model fine-tuning.

Responsibilities:

* initialize model
* run training loop
* track metrics
* save checkpoints
* save best model

### `evaluator.py`

Intrinsic evaluation of the router classifier.

Responsibilities:

* accuracy
* macro F1
* weighted F1
* per-class precision/recall/F1
* confusion matrix

### `diagnostics.py`

Generates debugging reports.

Responsibilities:

* label distribution
* confidence histogram
* misclassified examples
* example lengths
* source dataset distribution

---

## Training script

A standalone script should be available:

```text
scripts/train_router.py
```

Its responsibilities should be:

1. load training config
2. build or load processed dataset
3. load train/valid/test splits
4. initialize model
5. train router
6. evaluate router
7. save final artifact bundle

The script should support both:

* full preprocessing + training
* training from an already processed dataset

---

## Evaluation strategy

The router must be evaluated at two levels.

### 1. Intrinsic evaluation

This measures whether the classifier itself works.

Metrics:

* accuracy
* macro F1
* weighted F1
* per-class recall
* confusion matrix

This tells us whether the router can discriminate the four granularity classes.

### 2. Extrinsic evaluation

This measures whether the router improves the full chunking pipeline.

It should compare:

* no router
* fixed expert
* learned router
* oracle router

This is extremely important because a decent classifier may still fail to improve downstream chunking or retrieval.

---

## Required baselines

The router module should include at least the following baselines.

### Majority-class router

Always predicts the most frequent label.

Purpose:

* verify that the learned model beats trivial class imbalance exploitation

### Fixed expert router

Always selects one medium-granularity expert.

Purpose:

* compare against a practical no-learning baseline

### Length-based heuristic router

Uses only document length or simple heuristics.

Purpose:

* test whether the learned router is actually learning something beyond size

### Oracle router

Uses the true label derived from the reference chunks.

Purpose:

* estimate the upper bound of routing performance

---

## Inference design

The router inference logic must be separated from training.

### Runtime flow

At runtime, the router should perform the following sequence:

1. load artifact
2. preprocess document for inference
3. tokenize
4. predict probabilities
5. select label with `argmax`
6. map label to expert
7. return expert identifier

### Inference pseudocode

```text
function route_document(document):
    text = preprocess_for_inference(document)
    probabilities = router.predict_proba(text)
    label = argmax(probabilities)
    expert_id = label_to_expert(label)
    return expert_id
```

---

## Inference responsibilities by file

### `inference/router.py`

Main runtime router object used by the pipeline.

### `inference/loader.py`

Loads:

* model weights
* tokenizer
* config
* label map
* preprocessing metadata

### `inference/predictor.py`

Handles:

* tokenization
* forward pass
* probability extraction
* label prediction

### `inference/preprocessing.py`

Contains runtime input preparation.

This should be aligned with training-time preprocessing, but it does not have to be literally identical.

### `inference/mapping.py`

Contains explicit label-to-expert mapping.

### `inference/fallback.py`

Contains the fallback strategy.

---

## Label-to-expert mapping

This must be explicit and configurable.

Example:

```json
{
  "0": "expert_0_120",
  "1": "expert_120_150",
  "2": "expert_150_180",
  "3": "expert_180_inf"
}
```

This should not be hardcoded deep inside the model code.

### Why explicit mapping is important

Because later you may want to:

* rename experts
* swap experts
* compare different expert pools
* disable one expert temporarily

A saved mapping file makes all of this easier.

---

## Fallback behavior

Fallback behavior is essential and should be defined from the beginning.

The pipeline must specify what happens if:

* the artifact is missing
* the checkpoint cannot be loaded
* the prediction is invalid
* the confidence is too low
* the mapped expert is unavailable

### Recommended fallback options

#### Option A — fixed default expert

Use one expert defined in config.

#### Option B — middle-granularity expert

Use the most stable middle class.

#### Option C — confidence-aware fallback

If `max(probabilities) < threshold`, use the default expert.

This logic should be centralized in:

```text
src/router/inference/fallback.py
```

---

## Artifact design

A trained router must be saved as a self-contained bundle.

Recommended structure:

```text
artifacts/router/<router_name>/
  config.json
  label_map.json
  preprocess.json
  metrics.json
  tokenizer/
  checkpoints/
  best/
```

### What each file should contain

#### `config.json`

The effective training configuration.

#### `label_map.json`

The mapping between class ids and semantic meaning.

#### `preprocess.json`

The normalization strategy and related metadata.

#### `metrics.json`

Intrinsic evaluation results.

#### `tokenizer/`

The tokenizer used during training.

#### `best/`

The selected best checkpoint for inference.

---

## Configuration design

It is better to separate router configs from chunker configs.

### Example `configs/router/train.yaml`

```yaml
router:
  name: moc-router
  model_type: hf-sequence-classifier
  backbone: distilbert-base-uncased
  num_labels: 4

data:
  raw_input_path: data/router/raw/router_source.jsonl
  processed_output_path: data/router/processed/router_examples.jsonl
  train_path: data/router/splits/train.jsonl
  valid_path: data/router/splits/valid.jsonl
  test_path: data/router/splits/test.jsonl

labeling:
  bins: [120, 150, 180]

preprocessing:
  target_chars: 1024
  normalize_mode: chunk_aware

training:
  epochs: 3
  learning_rate: 1e-5
  train_batch_size: 3
  eval_batch_size: 3
  gradient_accumulation_steps: 16
  warmup_ratio: 0.1
  scheduler: cosine
  bf16: true

output:
  artifact_dir: artifacts/router/moc-router
```

### Example `configs/router/infer.yaml`

```yaml
router:
  enabled: true
  artifact_dir: artifacts/router/moc-router
  fallback_policy: default_expert
  default_label: 2
  confidence_threshold: null

mapping:
  0: expert_0_120
  1: expert_120_150
  2: expert_150_180
  3: expert_180_inf
```

---

## Diagnostics

The router module should save diagnostics from the first version.

Recommended diagnostics:

* label distribution
* source dataset distribution
* average normalized text length
* class imbalance report
* confusion matrix
* misclassified examples
* confidence histogram

These are critical because router failures are often hidden if you only look at overall accuracy.

For example:

* a router could collapse to one class
* a router could confuse only adjacent classes
* a router could be correct on paper but useless in downstream routing

Without diagnostics, these problems are hard to identify.

---

## Leakage and split safety

This is one of the most important implementation details.

### Possible leakage sources

* the same document appearing in both train and test
* near-duplicate documents across splits
* short-text concatenation mixing examples from different splits
* preprocessing done before split in a way that leaks document content

### Rules to avoid leakage

1. split by document identity
2. split before generating train-time mixtures
3. keep provenance metadata for every processed example
4. never concatenate material across split boundaries

This section should not be skipped.

A router trained with leakage may appear very strong while being useless in real deployment.

---

## Unit tests that should exist

The router module should include basic tests.

### Labeling tests

* correct behavior at boundaries
* no overlap between bins
* deterministic class assignment

### Normalization tests

* truncation works
* concatenation works
* chunk-aware behavior is respected

### Serialization tests

* processed dataset can be saved and loaded safely
* artifacts are complete

### Inference tests

* correct tensor shapes
* correct probability output format
* correct label-to-expert mapping
* fallback executes correctly

---

## Minimal viable implementation

The first working version does **not** need to solve everything.

A good MVP should include only:

* one processed router dataset
* one labeler
* one normalizer
* one HuggingFace classifier
* one training script
* one evaluation script
* one inference loader
* one label-to-expert mapping
* one fallback strategy

This is enough to validate the architecture before adding more complexity.

---

## Recommended implementation order

### Phase 1 — data foundations

1. define raw dataset schema
2. define processed dataset schema
3. implement `labeler.py`
4. implement `normalizer.py`
5. implement `dataset_builder.py`
6. generate the first processed dataset

### Phase 2 — first trainable router

7. implement `hf_sequence_classifier.py`
8. implement `collator.py`
9. implement `trainer.py`
10. implement `evaluator.py`
11. train the first baseline model
12. save artifact bundle

### Phase 3 — pipeline integration

13. implement `loader.py`
14. implement `predictor.py`
15. implement `mapping.py`
16. implement `fallback.py`
17. connect the router to the orchestrator

### Phase 4 — analysis

18. compare against fixed baselines
19. compare against oracle routing
20. inspect confusion matrix
21. evaluate downstream impact

---

## Practical recommendations

### Recommendation 1

Start with a **single backbone** and do not over-generalize too early.

### Recommendation 2

Keep the label generation logic fully isolated.

### Recommendation 3

Save all preprocessing metadata from day one.

### Recommendation 4

Implement fallback before pipeline integration.

### Recommendation 5

Run trivial baselines early, otherwise router improvements may be misleading.

### Recommendation 6

Do not mix training and inference code paths.

---

## Final design summary

The router should be implemented as a **standalone granularity classifier** with:

* a dedicated training pipeline
* a dedicated inference pipeline
* explicit label generation
* explicit preprocessing
* explicit mapping from labels to experts
* robust fallback behavior
* reproducible artifacts
* intrinsic and extrinsic evaluation

This gives a design that is:

* faithful to the MoC paper
* modular enough for the framework
* easy to test
* easy to extend later with new router variants

---

## Deliverables checklist

A complete router module should produce:

* raw router source dataset
* processed router dataset
* train/valid/test splits
* labeling logic
* normalization logic
* trainable router model
* saved artifact bundle
* intrinsic evaluation results
* diagnostics reports
* inference loader
* label-to-expert mapping
* fallback policy
* baseline comparisons

