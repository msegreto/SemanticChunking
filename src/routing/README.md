# Routing Module

Questa cartella ospita la fase di routing, che nel disegno del framework dovrebbe decidere come trattare le unità split prima del chunking.

File presenti:
- `base.py`: interfaccia astratta `BaseRouter`.
- `factory.py`: `RouterFactory`, usata dall'orchestrator per creare il router richiesto in configurazione; oggi espone gli alias `default` e `moc`.
- `dummy.py`: implementazione attuale. Non effettua un vero routing, ma propaga l'output aggiungendo `router_info`.
- `__init__.py`: abilita gli import del package.

Collegamenti:
- input: output di `src/splitting/`;
- output: struttura passata a `src/chunking/`;
- attivazione: controllata dalla sezione `router` del file experiment YAML.

Nel progetto attuale il routing è più un punto di estensione futura che una fase realmente implementata.
