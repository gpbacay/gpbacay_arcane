# FlyWire circuit snapshot

`flywire_escape_circuit.json` is a compact subgraph of the public FlyWire
female adult fly brain connectome (FAFB materialization 783).

Neuron metadata (root IDs, cell types, sides, soma coordinates, transmitters)
comes from the systematic annotations in
[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
(Schlegel et al., 2024). Synapse counts are from the public FlyWire connectivity
release (Dorkenwald et al., 2024; [Zenodo 10676866](https://zenodo.org/records/10676866)).

FlyWire data is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
This snapshot is a derived subset under the same terms: share and adapt with
attribution, non-commercial use only. It does not change the MIT license of
the ARCANE source code.

Please cite:

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature (2024). https://doi.org/10.1038/s41586-024-07558-y
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (2024). https://doi.org/10.1038/s41586-024-07686-5

Programmatic access uses [fafbseg-py](https://github.com/navis-org/fafbseg-py).
Refresh this snapshot with a CAVE token:

```
python examples/export_flywire_circuit.py --live
```
