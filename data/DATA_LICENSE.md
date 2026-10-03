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

# ARC 1 distillation data

The `arc1-tiny` checkpoint was trained with a frozen text encoder as teacher and
two public intent datasets (see `gpbacay_arcane/arc1_distill.py`). Neither the
datasets nor the teacher's vectors are redistributed in this repository; fetch
them yourself to retrain.

- **CLINC150** (Larson et al., 2019, *An Evaluation Dataset for Intent
  Classification and Out-of-Scope Prediction*),
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
  <https://github.com/clinc/oos-eval>. A fifth of its intents and everything
  resembling the held-out evaluation tools are excluded from training and used
  only for the real-utterance evaluation.
- **Banking77** (Casanueva et al., 2020, *Efficient Intent Detection with Dual
  Sentence Encoders*), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  <https://github.com/PolyAI-LDN/task-specific-datasets>.
- **Qwen3-Embedding-0.6B** (Qwen Team, 2025),
  [Apache 2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): the teacher
  whose embeddings ARC 1 learns to match. It is not part of the model and is not
  needed at inference.

```
mkdir -p data/external
curl -L https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json -o data/external/clinc150_data_full.json
curl -L https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv -o data/external/banking77_train.csv
```

The ARC 1 weights are a derivative of these datasets, so keep this attribution
with any redistribution of them. The MIT license of the ARCANE source code is unchanged.
