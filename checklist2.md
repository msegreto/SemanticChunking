# Semantic Clustering Chunking Checklist

## Obiettivo

Pianificare l'implementazione del chunker `semantic_clustering` nel framework corrente, usando come riferimento principale il paper "Is Semantic Chunking Worth the Computational Cost?" e mantenendo compatibilita' con l'architettura gia' usata da `semantic_breakpoint`.

## Paper Notes

### Cosa valuta il paper

- Confronta tre famiglie di chunking: `fixed-size`, `semantic breakpoint`, `semantic clustering`.
- Tutti i documenti vengono prima splittati in frasi con `SpaCy en_core_web_sm`.
- I chunker vengono valutati indirettamente su tre task:
  - document retrieval
  - evidence retrieval
  - retrieval-based answer generation
- La metrica principale per retrieval e' `F1@k`, non solo recall, per bilanciare la tendenza dei chunk grandi ad aumentare recall ma introdurre rumore.
- Per answer generation usa `BERTScore`; il paper osserva differenze molto piccole tra i chunker.

### Definizione del semantic clustering nel paper

- Il clustering-based chunker lavora a livello frase.
- Puo' raggruppare frasi non contigue.
- Usa una distanza congiunta posizione + semantica:
  - `d(x_a, x_b) = lambda * d_pos(x_a, x_b) + (1 - lambda) * d_cos(x_a, x_b)`
  - `d_pos(x_a, x_b) = |a - b| / n`
  - `d_cos(x_a, x_b) = 1 - max(cos(emb(x_a), emb(x_b)), 0)`
- `lambda = 0` significa chunking puramente semantico.
- `lambda = 1` degenera verso un comportamento simile al fixed-size.
- Le similarita' cosine negative vengono clampate a `0` prima di convertirle in distanza.

### Algoritmi usati nel paper

- `single-linkage agglomerative clustering`
- `DBSCAN`

### Adjustments espliciti riportati dal paper

- Sul single-linkage introducono un `maximum chunk size` per evitare che quasi tutto il documento collassi in un chunk unico.
- Quel vincolo dipende da:
  - numero di chunk desiderati
  - numero totale di frasi nel documento
- Introducono anche una `distance threshold for stopping`.
- Se la distanza minima supera la soglia, il clustering si ferma e le frasi rimanenti restano non raggruppate.
- Nel paper questa soglia di stop e' fissata a `0.5`.

### Hyperparameter space del paper

- Single-linkage:
  - `lambda in [0, 0.25, 0.5, 0.75, 1]`
  - `n_chunks in [2..10]`
- DBSCAN:
  - `lambda in [0, 0.25, 0.5, 0.75, 1]`
  - `eps in [0.1, 0.2, 0.3, 0.4, 0.5]`
  - `min_samples in [1, 2, 3, 4, 5]`

### Risultati e takeaways principali

- I guadagni del semantic chunking non sono consistenti abbastanza da giustificare in generale il costo computazionale.
- Il `semantic breakpoint` va meglio soprattutto su dataset sintetici "stitched", dove documenti corti e topic diversi vengono concatenati artificialmente.
- Su documenti lunghi piu' realistici, il `fixed-size` e' spesso uguale o migliore.
- In evidence retrieval e answer generation le differenze tra metodi sono piccole.
- L'embedder ha un impatto forte; nel paper conta piu' del chunker in molti casi.
- Per il clustering, il paper osserva che aumentare il peso della posizione (`lambda -> 1`) tende a migliorare F1 nell'evidence retrieval.
- Il paper conclude in pratica: fixed-size resta la scelta piu' affidabile per applicazioni RAG standard.

### Limiti importanti del paper

- Tutto e' sentence-level: le embedding di frase possono perdere contesto locale.
- I task sono proxy task: non c'e' ground truth chunk-level.
- Parte del vantaggio dei semantic chunker emerge su documenti sintetici stitched, quindi non e' detto che generalizzi bene.
- Il clustering puo' mischiare frasi lontane ma semanticamente simili, producendo chunk poco utili per retrieval sequenziale.

## Implicazioni Per Questo Repo

- Abbiamo gia' la base tecnica giusta:
  - `src/chunking/semantic_base.py`
  - `src/chunking/semantic_utils.py`
  - `src/chunking/semantic_breakpoint.py`
  - `src/chunking/semantic_clustering.py` come placeholder
- Il framework e' gia' document-centric e sentence-level, quindi aderisce bene all'impostazione del paper.
- La scelta pragmatica per una prima implementazione e' partire da `single-linkage`, non da `DBSCAN`.
- Motivo:
  - e' piu' vicino alla formulazione principale del paper
  - richiede meno superficie di configurazione
  - permette confronto diretto con `fixed` tramite `n_chunks`
  - riduce il rischio di introdurre una prima versione troppo instabile
- Inferenza per il progetto:
  - conviene implementare prima una variante paper-faithful e solo dopo decidere se estendere a DBSCAN
  - conviene impedire, almeno all'inizio, chunk non contigui nell'output finale oppure renderli espliciti nei metadata e nella valutazione
- Nota critica:
  - il paper permette cluster non contigui, ma il resto della pipeline sembra assumere chunk con testo aggregato e ordine lineare; questa tensione va risolta in design prima di scrivere l'algoritmo.

## Decisioni Di Design Da Fissare

- [ ] Decidere se il chunker deve produrre:
  - chunk non contigui fedeli al paper
  - oppure span contigui derivati dai cluster come adattamento per la pipeline
- [ ] Decidere la definizione esatta di `maximum chunk size`:
  - hard cap sul numero di frasi per cluster
  - derivata da `ceil(num_sentences / n_chunks_target)`
- [ ] Decidere se esporre la `stop_distance_threshold` come config o fissarla inizialmente a `0.5` come nel paper
- [ ] Decidere se l'MVP supporta solo `single_linkage` oppure anche `dbscan`

## MVP Proposto

- [ ] Implementare `single_linkage` soltanto nella prima iterazione
- [ ] Supportare i parametri:
  - `lambda`
  - `number_of_chunks`
  - `max_chunk_size` opzionale oppure derivato
  - `stop_distance_threshold` default `0.5`
- [ ] Mantenere `similarity_metric: cosine`
- [ ] Riutilizzare l'embedder gia' gestito da `BaseSemanticChunker`
- [ ] Salvare metadata completi per cache compatibility e reproducibility
- [ ] Aggiungere poi `dbscan` solo se l'MVP produce risultati sensati

## Implementazione Tecnica

### 1. Config e validazione

- [ ] Sostituire il placeholder in `src/chunking/semantic_clustering.py`
- [ ] Validare `clustering_mode` con valore iniziale ammesso: `single_linkage`
- [ ] Validare `lambda` come float in `[0, 1]`
- [ ] Validare `number_of_chunks` come int positivo
- [ ] Validare `stop_distance_threshold` come float `>= 0`
- [ ] Validare `max_chunk_size` come int positivo oppure calcolarlo automaticamente
- [ ] Aggiornare metadata builder e matcher per tutti i nuovi campi

### 2. Utility numeriche

- [ ] Aggiungere in `src/chunking/semantic_utils.py` una utility per:
  - cosine distance clampata con `max(similarity, 0)`
- [ ] Aggiungere una utility per costruire la joint distance matrix:
  - positional distance normalizzata
  - semantic distance
  - combinazione pesata da `lambda`
- [ ] Valutare se serve una utility per ricostruire cluster ordinati per `sentence_idx`

### 3. Algoritmo single-linkage

- [ ] Embeddare tutte le frasi ordinate del documento una sola volta
- [ ] Costruire la matrice delle distanze pairwise
- [ ] Implementare il merge greedy single-linkage
- [ ] Fermare i merge quando:
  - il merge violerebbe `max_chunk_size`
  - oppure la minima distanza supera `stop_distance_threshold`
- [ ] Gestire le frasi non raggruppate come cluster singoli finali
- [ ] Ordinare i cluster finali in base alla prima frase del cluster

### 4. Adattamento output al framework

- [ ] Definire come serializzare un cluster nel formato chunk del repo
- [ ] Se il cluster contiene frasi non contigue, decidere se:
  - concatenarle nell'ordine originale
  - oppure convertirle in span contigui separati
- [ ] Garantire che i campi richiesti dalla pipeline restino coerenti:
  - `doc_id`
  - testo chunk
  - posizione chunk
  - metadati di provenienza frase
- [ ] Verificare che retrieval e intrinsic evaluation non assumano implicitamente contiguita'

### 5. Documentazione

- [ ] Aggiornare `src/chunking/README.md`

## Ordine Di Esecuzione Consigliato

- [ ] Fase 1: fissare decisione su contiguita' dei chunk
- [ ] Fase 2: implementare validazione config + metadata
- [ ] Fase 3: implementare utilities per distance matrix
- [ ] Fase 4: implementare single-linkage MVP
- [ ] Fase 5: aggiungere test unitari
- [ ] Fase 6: aggiungere YAML esperimento
- [ ] Fase 7: eseguire benchmark iniziale contro fixed e breakpoint
- [ ] Fase 8: decidere se DBSCAN vale la pena

## Raccomandazione Operativa

- [ ] Non inseguire subito tutto il paper.
- [ ] Implementare prima un `semantic_clustering(single_linkage)` minimale, riproducibile e misurabile.
- [ ] Trattare `DBSCAN` come fase successiva.
- [ ] Misurare esplicitamente il costo computazionale del chunker, altrimenti rischiamo di ripetere l'errore evidenziato dal paper: aggiungere complessita' senza un guadagno stabile.

## Riferimenti

- Paper principale: https://aclanthology.org/2025.findings-naacl.114/
- PDF: https://aclanthology.org/2025.findings-naacl.114.pdf
