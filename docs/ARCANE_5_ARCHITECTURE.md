# ARCANE 5.0 architecture

ARCANE 5.0 focuses the project on one purpose: making System 1 AI models through adaptive inference, grounded decisions, neural memory, and biologically inspired model components.

In ARCANE, “System 1” is a measurable software contract for fast, automatic, bounded decisions. It is not a claim that a model is conscious, human-equivalent, or a literal simulation of a brain.

## The contract

A deployed ARCANE System 1 model follows one loop:

1. **Perceive** an input and its allowed decision space.
2. **Recall** similar decided examples from neural memory.
3. **Adapt** transient state or selected fast weights within explicit limits when requested.
4. **Decide** inside the declared schema, reporting confidence, evidence, and an abstention when thresholds are not met.

`System1.decide()` returns a `GroundedDecision` containing:

- `label`: a member of the `DecisionSpace`, or `None`;
- `confidence` and the complete normalized `distribution`;
- `abstained` and a machine-readable `reason`;
- `source`: model, memory, adaptive model, or a composition of them;
- `evidence`: recalled examples that influenced the decision;
- `adaptation`: objective and update diagnostics when test-time adaptation ran;
- `latency_ms` and backend metadata.

Outputs outside the declared decision space are discarded and reported under `metadata.ignored_labels`. Invalid, negative, or non-finite scores cannot pass grounding.

## The four pillars

| Pillar | ARCANE 5.0 public surface | Existing mechanisms used |
| --- | --- | --- |
| Adaptive inference | `AdaptiveInference`, `AdaptationTrace` | `TestTimeAdapter`, resonance cycles, feedback alignment, BCM fast weights, rollback and trust regions |
| Grounded decisions | `DecisionSpace`, `GroundedDecision`, `System1` | ARC 1 classification, extraction and tool schemas; explicit confidence and abstention |
| Neural memory | `NeuralMemory`, `System1.remember()` | `Hippocampus`, `DocumentGraph`, weighted example recall |
| Biological components | Standalone layers and full models | GSER/ResonantGSER, predictive resonance, Hebbian/BCM plasticity, homeostasis, ConceptEngram |

Grounded Graph of Thought remains the evidence-verification path for jobs that require documents and multi-step language reasoning. ARC 1 remains the compact direct-decision model. The causal SLM remains available, but it is no longer the project's primary entry point.

## Minimal composition

```python
from gpbacay_arcane import DecisionSpace, System1

def predictor(text, labels):
    # Replace with ARC 1, a trained classifier, or another direct model.
    return {label: 1.0 if label in text.lower() else 0.1 for label in labels}

model = System1(
    predictor,
    DecisionSpace(["approve", "review", "decline"], min_confidence=0.6),
)
model.remember("manual review required for incomplete ID", "review")
decision = model.decide("incomplete ID submitted")
```

For a Keras model, create an `AdaptiveInference(model, variables=...)` and pass it as `adapter=`. Calling `decide(..., adapt=True, adaptation_inputs=batch)` uses the adapter's predictions and attaches its objective, accepted-step, rollback, and update-norm trace. Adaptation stays episodic by default.

## Compatibility and migration

ARCANE 5.0 is additive at the Python API level:

- Existing 4.x imports, layers, model wrappers, ARC 1 APIs, and saved-model utilities remain available.
- `TestTimeAdapter` and `Hippocampus` keep their names; `AdaptiveInference` and `NeuralMemory` are clearer 5.0 entry points built on them.
- Code that already calls `Hippocampus.classify()` can continue doing so. Move to `System1` when it needs a uniform abstention, provenance, and adaptation contract.
- Generative SLM functionality remains supported for open-ended language work, but should be treated as a separate generative path rather than the default System 1 output.

The package version, Node package version, website positioning, examples, and CLI identity are aligned at `5.0.0`.

## Evaluation requirements

Calling a model “System 1” requires evidence on its target task. Report these dimensions separately:

- decision quality and calibration;
- abstention coverage and selective risk;
- latency and compute per decision;
- adaptation gain, update norm, rollback rate, and order sensitivity;
- memory gain, harmful-recall rate, and provenance coverage;
- ablations for each biological component;
- distribution-shift and forgetting behavior.

The “first library” language is project positioning, not experimental evidence. Individual mechanisms should only be credited for improvements demonstrated by controlled ablations from the same checkpoint.
