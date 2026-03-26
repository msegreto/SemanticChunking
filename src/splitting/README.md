# Splitting Module

Questa cartella implementa la fase di splitting, cioè la trasformazione dei documenti normalizzati in unità testuali più piccole che possono poi essere chunkate.

File presenti:
- `base.py`: interfaccia astratta `BaseSplitter` (API streaming).
- `factory.py`: `SplitterFactory`, usata dall'orchestrator per selezionare lo splitter dal YAML.
- `sentence.py`: implementazione principale nel flusso corrente. Espone i metodi incrementali `build_streaming_components(...)` e `split_document_streaming(...)`.
- `contextualizer.py`: splitter sentence-compatible che applica un passaggio di context rewriting (via modulo `context_resolver`) mantenendo lo stesso schema output del sentence splitter.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: dati normalizzati prodotti da `src/datasets/`;
- output: struttura consumata da `src/routing/` o direttamente da `src/chunking/`;
- persistenza: quando `split.save_output` è abilitato, il pipeline salva nel path configurato (`data/processed/` o `data/processed_context/`).

Nel flusso attuale sono supportati in streaming `type: sentence` e `type: contextualizer`.

Configurazione `type: contextualizer`:
- usa i campi sotto `split.context_resolver`;
- `backend` supportato: `hf_remote`.
- quando `enabled: true`, endpoint obbligatorio (`endpoint_url` o `endpoint_url_env`).
- prompt supportato:
  - `prompt_template`: template con placeholder `{text}`;
  - `max_input_chars`: limite di sicurezza sul testo inviato al modello.
