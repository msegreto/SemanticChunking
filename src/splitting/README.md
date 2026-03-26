# Splitting Module

Questa cartella implementa la fase di splitting come stage di pipeline. L'unica API pubblica e' [`SplitTransformer`](/Users/mattiasegreto/Desktop/TesiCode/src/splitting/transformer.py), che costruisce internamente la variante richiesta da `split.type`.

Struttura:
- `transformer.py`: entrypoint pubblico. Espone `SplitTransformer.from_config(...)` e gestisce cache, persistenza e scelta dello splitter.
- `sentence.py`: implementazione interna per `split.type: sentence`.
- `contextualizer.py`: implementazione interna per `split.type: contextualizer`.
- `context_resolver/`: supporto interno al contextualizer.

Contratto:
- input: corpus normalizzato PyTerrier-friendly con documenti `{"docno", "text", ...}`
- output: artifact offline di split units JSONL
- schema split unit: `unitno`, `parent_docno`, `position`, `text`, `char_start`, `char_end`

Configurazioni supportate:
- `split.type: sentence`
- `split.type: contextualizer`

Per `split.type: contextualizer`:
- la configurazione vive sotto `split.context_resolver`
- `backend` supportato: `hf_remote`
- se `enabled: true`, serve `endpoint_url` oppure `endpoint_url_env`
- sono supportati `prompt_template` e `max_input_chars`
