# Intrinsic Metrics

Questa cartella contiene le metriche elementari e le utility usate dalla valutazione intrinsic.

File presenti:
- `boundary_clarity.py`: misura quanto il confine tra chunk consecutivi sia "chiaro" secondo il rapporto tra perplexity semplice e condizionata.
- `chunk_stickiness.py`: misura quanto i chunk risultino coesi o collegati tra loro tramite una costruzione ispirata a edge weights e structural entropy.
- `utils.py`: normalizza il formato del `chunk_output` e conta i chunk totali.
- `stats.py`: produce statistiche aggregate globali a partire dalle metriche per documento.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- vengono richiamate da `src/evaluation/intrinsic/default.py`;
- usano lo scorer definito in `src/evaluation/intrinsic/models/perplexity_scorer.py`;
- i risultati finali vengono salvati in `results/intrinsic/`.
