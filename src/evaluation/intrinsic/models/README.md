# Intrinsic Models

Questa cartella contiene i modelli o scorer ausiliari usati dalla valutazione intrinsic.

File presenti:
- `perplexity_scorer.py`: astrazione centrale per calcolare `perplexity(text)` e `conditional_perplexity(target | context)`. Supporta un backend lessicale leggero e un backend basato su Hugging Face causal LM.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- viene usato dalle metriche in `src/evaluation/intrinsic/metrics/`;
- viene configurato tramite la sezione `evaluation.intrinsic_model` del file experiment YAML.

Questa cartella è importante perché controlla il costo computazionale e il significato delle metriche intrinsic.
