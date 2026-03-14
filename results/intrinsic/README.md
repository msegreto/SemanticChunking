# Intrinsic Results

Questa cartella contiene i risultati della valutazione intrinsic dei chunk.

File osservati:
- `intrinsic_fiqa_intrinsic_results.json`: risultati dettagliati per documento e metriche aggregate;
- `intrinsic_fiqa_intrinsic_results_global.json`: statistiche globali derivate dai risultati per documento.

Questi file sono scritti da `src/evaluation/intrinsic/default.py`.

Sono il punto finale della parte intrinsic del pipeline e servono per capire come si comporta una strategia di chunking prima ancora di passare al retrieval.
