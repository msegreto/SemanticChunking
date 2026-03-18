# Evaluation Profiles

Questa cartella contiene profili YAML riusabili per la sezione `evaluation`.

Uso consigliato in un file `configs/experiments/*.yaml`:

```yaml
evaluation_profile: default

evaluation:
  extrinsic_tasks_to_run: [document_retrieval, evidence_retrieval, answer_generation]
  intrinsic_model:
    backend: lexical
```

Il merge e' deep-merge:
- base = profilo (`configs/evaluations/<nome>.yaml`);
- override = blocco `evaluation` del file esperimento.
