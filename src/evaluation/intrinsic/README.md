# Intrinsic Evaluation

Questa cartella contiene la valutazione intrinsic della qualità dei chunk, cioè metriche che operano direttamente sulla struttura e sul contenuto dei chunk senza passare da un task esterno.

File presenti:
- `base.py`: interfaccia `BaseIntrinsicEvaluator`.
- `factory.py`: factory che istanzia l'evaluator richiesto.
- `default.py`: implementazione principale. Normalizza l'output del chunker, calcola le metriche abilitate, aggrega i risultati e li salva in `results/intrinsic/`.
- `metrics/`: implementazioni delle metriche e utility statistiche.
- `models/`: scorer usato dalle metriche per calcolare perplexity o conditional perplexity.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: `chunk_output` prodotto da `src/chunking/`;
- dipendenze: metriche in `metrics/` e scorer in `models/`;
- output: JSON salvati in `results/intrinsic/`.

Nel baseline attuale vengono calcolate `boundary_clarity` e `chunk_stickiness`.
