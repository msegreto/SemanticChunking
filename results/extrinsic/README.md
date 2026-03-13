# Extrinsic Results

Questa cartella contiene i risultati della valutazione extrinsic, oggi focalizzata sul document retrieval.

File osservati:
- `beir-fiqa_fixed_no-routing_baseline_pipeline_base.csv`: CSV del baseline su FIQA con chunking `fixed`, nessun routing, embedder MPNet e retriever NumPy.

Questo file viene prodotto da `scripts/run_extrinsic_eval.py` usando i moduli in `src/evaluation/extrinsic/`.

Le colonne del CSV includono informazioni sia di performance sia di tracciabilità:
- dataset, chunking, routing, retriever, embedder;
- valori di `k`;
- precision, recall, F1;
- path usati per query, qrels e indice.
