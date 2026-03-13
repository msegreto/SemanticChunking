# Evaluation Module

Questa cartella raccoglie tutta la logica di valutazione del framework.

La valutazione è divisa in due famiglie:
- `intrinsic/`: misura proprietà dei chunk prodotti;
- `extrinsic/`: misura l'impatto dei chunk su task downstream, oggi soprattutto document retrieval.

File presenti:
- `__init__.py`: abilita gli import del package.
- `intrinsic/`: evaluator, metriche e scorer per analizzare i chunk.
- `extrinsic/`: evaluator, IO e metriche per retrieval.

Collegamenti:
- l'orchestrator invoca prima la parte intrinsic direttamente;
- la parte extrinsic viene poi richiamata tramite `scripts/run_extrinsic_eval.py`;
- gli output finiscono in `results/intrinsic/` e `results/extrinsic/`.
