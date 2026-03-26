# Intrinsic Evaluation

Questa cartella contiene la valutazione intrinsic della qualità dei chunk come stage di pipeline. L'unica API pubblica e' [`IntrinsicEvaluationTransformer`](/Users/mattiasegreto/Desktop/TesiCode/src/evaluation/intrinsic/transformer.py).

Struttura:
- `transformer.py`: entrypoint pubblico. Calcola le metriche intrinsic e salva i risultati.
- `metrics/`: implementazioni delle metriche e utility statistiche.
- `models/`: scorer usato dalle metriche per calcolare perplexity o conditional perplexity.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: `chunk_output` prodotto da `src/chunking/`;
- dipendenze: metriche in `metrics/` e scorer in `models/`;
- output: JSON salvati in `results/intrinsic/`.

Nel baseline attuale vengono calcolate `boundary_clarity` e `chunk_stickiness`.

## Nota su `sequential_delta` (Chunk Stickiness)

La metrica `chunk_stickiness_incomplete` usa il vincolo sequenziale:
- aggiunge un arco solo se `j - i > sequential_delta`.

Implicazione pratica:
- con `sequential_delta: 0`, dato che le coppie considerate sono gia' `j > i`, il grafo incomplete puo' coincidere con il complete;
- con `sequential_delta: 1` (default consigliato nei nostri YAML), il grafo incomplete esclude almeno gli archi tra chunk adiacenti, riducendo la degenerazione `CS_i == CS_c`.

Quindi:
- `sequential_delta: 0` e' utile solo per una lettura strict-text del paper;
- `sequential_delta: 1` e' piu' utile per analisi sperimentale, perche' rende `CS_i` piu' informativa.
