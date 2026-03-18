# Execution Profiles

Questa cartella contiene profili YAML riusabili per la sezione `execution`.

Uso consigliato in un file `configs/experiments/*.yaml`:

```yaml
execution_profile: default

execution:
  streaming_docs_per_run: 100
```

Il merge e' deep-merge:
- base = profilo (`configs/execution/<nome>.yaml`);
- override = blocco `execution` del file esperimento.
