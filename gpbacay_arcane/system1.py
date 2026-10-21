"""The ARCANE 5.0 composition API for System 1 decision models.

The lower-level ARCANE modules remain independently useful.  This module gives
them one deployment contract: perceive an input, optionally adapt on it, recall
similar decided examples, and return a bounded decision instead of unconstrained
generated text. Tool calls use the same confidence and margin bar before a
handler runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hippocampus import Hippocampus
from .test_time_adaptation import TestTimeAdapter
from .tools import commit_tools


@dataclass(frozen=True)
class AdaptationTrace:
    """Serializable diagnostics for one bounded inference-time adaptation."""

    baseline_objective: float
    final_objective: float
    best_objective: float
    attempted_steps: int
    accepted_steps: int
    rolled_back: bool
    update_norms: Tuple[float, ...] = ()

    @classmethod
    def from_result(cls, result: Any) -> "AdaptationTrace":
        return cls(
            baseline_objective=float(result.baseline_objective),
            final_objective=float(result.final_objective),
            best_objective=float(result.best_objective),
            attempted_steps=int(result.attempted_steps),
            accepted_steps=int(result.accepted_steps),
            rolled_back=bool(result.rolled_back),
            update_norms=tuple(float(value) for value in result.update_norms),
        )


@dataclass(frozen=True)
class GroundedDecision:
    """A decision constrained to a declared space, with uncertainty and provenance."""

    label: Optional[str]
    confidence: float
    distribution: Mapping[str, float]
    abstained: bool
    reason: Optional[str]
    source: str
    evidence: Tuple[Mapping[str, Any], ...] = ()
    adaptation: Optional[AdaptationTrace] = None
    latency_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""
        value = asdict(self)
        value["distribution"] = dict(self.distribution)
        value["evidence"] = [dict(item) for item in self.evidence]
        value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True)
class DecisionSpace:
    """The only decisions a System 1 model is allowed to make.

    ``min_confidence`` creates an explicit abstention boundary. ``min_margin``
    additionally rejects ambiguous results whose two leading choices are too
    close. Model outputs outside ``labels`` are never allowed through.
    """

    labels: Tuple[str, ...]
    min_confidence: float = 0.5
    min_margin: float = 0.0

    def __init__(
        self,
        labels: Sequence[str],
        min_confidence: float = 0.5,
        min_margin: float = 0.0,
    ):
        normalized = tuple(str(label) for label in labels)
        if not normalized or any(not label.strip() for label in normalized):
            raise ValueError("DecisionSpace needs at least one non-empty label")
        if len(set(normalized)) != len(normalized):
            raise ValueError("DecisionSpace labels must be unique")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1")
        object.__setattr__(self, "labels", normalized)
        object.__setattr__(self, "min_confidence", float(min_confidence))
        object.__setattr__(self, "min_margin", float(min_margin))

    def ground(
        self,
        scores: Mapping[str, float],
        *,
        source: str = "model",
        evidence: Sequence[Mapping[str, Any]] = (),
        adaptation: Optional[AdaptationTrace] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GroundedDecision:
        """Normalize scores inside the schema and apply abstention policy."""
        unknown = sorted(set(scores) - set(self.labels))
        values: Dict[str, float] = {}
        for label in self.labels:
            value = float(scores.get(label, 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Score for {label!r} must be finite and non-negative")
            values[label] = value

        total = sum(values.values())
        distribution = (
            {label: value / total for label, value in values.items()}
            if total > 0.0
            else {label: 0.0 for label in self.labels}
        )
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], self.labels.index(item[0])))
        confidence = ranked[0][1] if ranked else 0.0
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        reason = None
        if total <= 0.0:
            reason = "no_supported_decision"
        elif confidence < self.min_confidence:
            reason = "low_confidence"
        elif confidence - runner_up < self.min_margin:
            reason = "ambiguous_decision"
        abstained = reason is not None
        details = dict(metadata or {})
        if unknown:
            details["ignored_labels"] = unknown
        return GroundedDecision(
            label=None if abstained else ranked[0][0],
            confidence=confidence,
            distribution=distribution,
            abstained=abstained,
            reason=reason,
            source=source,
            evidence=tuple(dict(item) for item in evidence),
            adaptation=adaptation,
            latency_ms=float(latency_ms),
            metadata=details,
        )


class _SilentTools:
    """Memory-only tool pass: propose nothing, and let remembered patterns fill calls."""

    def run(self, prompt, tools=None, tool_prior=None, execute=False, **kwargs):
        del prompt, tools, execute, kwargs
        return {"function_calls": [], "confidence": None, "source": "memory", "tool_prior": tool_prior}


def _tool_distribution(out: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    decisions = out.get("decisions")
    scores = decisions.get("tools") if isinstance(decisions, dict) else None
    if not isinstance(scores, dict):
        return None
    return {
        str(name): float(score)
        for name, score in scores.items()
        if isinstance(score, (int, float)) and not isinstance(score, bool) and np.isfinite(score)
    } or None


class AdaptiveInference(TestTimeAdapter):
    """ARCANE 5.0 name for bounded, measurable, reversible test-time adaptation."""


class NeuralMemory(Hippocampus):
    """ARCANE 5.0 name for fast episodic memory of previously decided examples."""


class System1:
    """Compose prediction, bounded adaptation, decision memory, and grounding.

    Args:
        predictor: A callable ``(input, labels, **kwargs) -> scores``; an object
            with ``classify(input, labels, **kwargs)``; or an object with
            ``predict(input, **kwargs)``. It may be ``None`` for memory-only use.
            ``act`` requires ``run(prompt, tools=, tool_prior=, execute=)``.
        decisions: The model's closed output space and abstention policy.
        memory: Optional ``NeuralMemory``/``Hippocampus``. A new memory is
            created when omitted.
        adapter: Optional ``AdaptiveInference`` or compatible object with an
            ``adapt`` method. Its adapted predictions are used directly.
        memory_weight: Blend assigned to recalled decisions when both the model
            and memory have evidence.
        scores_are_logits: Apply softmax to numeric predictor arrays.
    """

    def __init__(
        self,
        predictor: Any,
        decisions: DecisionSpace,
        *,
        memory: Optional[Hippocampus] = None,
        adapter: Optional[TestTimeAdapter] = None,
        memory_weight: float = 0.5,
        scores_are_logits: bool = False,
    ):
        if not isinstance(decisions, DecisionSpace):
            raise TypeError("decisions must be a DecisionSpace")
        if not 0.0 <= memory_weight <= 1.0:
            raise ValueError("memory_weight must be between 0 and 1")
        self.predictor = predictor
        self.decisions = decisions
        self.memory = memory if memory is not None else NeuralMemory()
        self.adapter = adapter
        self.memory_weight = float(memory_weight)
        self.scores_are_logits = bool(scores_are_logits)

    @staticmethod
    def _normalize(scores: Mapping[str, float]) -> Dict[str, float]:
        values = {label: max(0.0, float(value)) for label, value in scores.items()}
        total = sum(values.values())
        return {label: value / total for label, value in values.items()} if total else values

    def _array_scores(self, output: Any) -> Dict[str, float]:
        array = np.asarray(output, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1 or array.shape[0] != len(self.decisions.labels):
            raise ValueError(
                "Numeric predictions for one decision must have shape "
                f"({len(self.decisions.labels)},) or (1, {len(self.decisions.labels)})"
            )
        if self.scores_are_logits:
            shifted = array - np.max(array)
            array = np.exp(shifted) / np.sum(np.exp(shifted))
        return dict(zip(self.decisions.labels, (float(value) for value in array)))

    def _coerce_scores(self, output: Any) -> Tuple[Dict[str, float], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        if isinstance(output, Mapping):
            if "distribution" in output:
                metadata = {
                    key: value
                    for key, value in output.items()
                    if key not in {"distribution", "label", "confidence", "neighbors"}
                }
                output = output["distribution"]
            return {str(key): float(value) for key, value in output.items()}, metadata
        return self._array_scores(output), metadata

    def _predict(self, input_data: Any, kwargs: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if self.predictor is None:
            return {}, {}
        labels = list(self.decisions.labels)
        if hasattr(self.predictor, "classify"):
            output = self.predictor.classify(input_data, labels, **kwargs)
        elif hasattr(self.predictor, "predict"):
            output = self.predictor.predict(input_data, **kwargs)
        elif callable(self.predictor):
            output = self.predictor(input_data, labels, **kwargs)
        else:
            raise TypeError("predictor must be callable or expose classify() or predict()")
        return self._coerce_scores(output)

    def decide(
        self,
        input_data: Any,
        *,
        cue: Optional[str] = None,
        adapt: bool = False,
        adaptation_inputs: Any = None,
        adaptation_kwargs: Optional[Mapping[str, Any]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> GroundedDecision:
        """Run the System 1 loop and return one grounded, inspectable decision.

        ``cue`` is the text used for episodic recall. It defaults to
        ``input_data`` when that input is a string. Tensor models can therefore
        pass encoded inputs as ``input_data`` and their original text as ``cue``.
        """
        started = perf_counter()
        adaptation = None
        metadata: Dict[str, Any] = {}
        if adapt:
            if self.adapter is None:
                raise ValueError("adapt=True requires an AdaptiveInference adapter")
            adapted = self.adapter.adapt(
                input_data if adaptation_inputs is None else adaptation_inputs,
                **dict(adaptation_kwargs or {}),
            )
            scores = self._array_scores(adapted.predictions)
            adaptation = AdaptationTrace.from_result(adapted)
            source = "adaptive-model"
        else:
            scores, metadata = self._predict(input_data, dict(model_kwargs or {}))
            source = "model"

        recall_cue = cue if cue is not None else (input_data if isinstance(input_data, str) else None)
        evidence = []
        if recall_cue:
            recalled = self.memory.react(recall_cue, self.decisions.labels)
            votes, evidence = recalled["votes"], recalled["neighbors"]
            if votes:
                memory_scores = {label: float(votes.get(label, 0.0)) for label in self.decisions.labels}
                if scores:
                    model_scores = self._normalize(scores)
                    scores = {
                        label: (1.0 - self.memory_weight) * model_scores.get(label, 0.0)
                        + self.memory_weight * memory_scores.get(label, 0.0)
                        for label in self.decisions.labels
                    }
                    source += "+memory"
                else:
                    scores, source = memory_scores, "memory"

        if not scores and self.predictor is None:
            source = "memory"
        return self.decisions.ground(
            scores,
            source=source,
            evidence=evidence,
            adaptation=adaptation,
            latency_ms=(perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def remember(self, cue: str, label: Optional[str], arguments: Optional[Dict[str, Any]] = None) -> str:
        """Teach one decided example immediately, without updating model weights."""
        if label is not None and label not in self.decisions.labels:
            raise ValueError(f"{label!r} is outside the decision space")
        return self.memory.remember(cue, label, arguments)

    def observe(self, cue: str, label: Optional[str], arguments: Optional[Dict[str, Any]] = None,
                trust: float = 1.0) -> Optional[str]:
        """Learn from an outcome: the memory keeps it only if it is surprising and trusted enough."""
        if label is not None and label not in self.decisions.labels:
            raise ValueError(f"{label!r} is outside the decision space")
        return self.memory.observe(cue, label, arguments, trust, labels=self.decisions.labels)

    def forget(self, cue: str) -> bool:
        """Remove a previously remembered example."""
        return self.memory.forget(cue)

    def act(
        self,
        prompt: str,
        tools: Sequence[Any],
        *,
        execute: bool = True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Choose tool calls, then run a handler only when the call clears ``decisions``.

        The predictor implements ``run(prompt, tools=, tool_prior=, execute=)`` and returns
        ``function_calls`` and ``confidence``. ``None`` uses memory alone: a remembered
        pattern can still propose a call, and a missing score holds it. A predictor with
        no ``run`` method cannot take this path.
        """
        started = perf_counter()
        if self.predictor is not None and not hasattr(self.predictor, "run"):
            raise TypeError(
                "act() needs a predictor with run(prompt, tools=, tool_prior=, execute=); "
                "use decide() for labels"
            )
        runner = self.predictor if self.predictor is not None else _SilentTools()
        out = dict(self.memory.run(
            prompt, tools=list(tools), execute=False, model=runner, **dict(model_kwargs or {}),
        ))
        if self.predictor is None and out.get("source") == "model+memory":
            out["source"] = "memory"
        decision = commit_tools(
            out.get("function_calls") or [],
            list(tools),
            out.get("confidence"),
            min_confidence=self.decisions.min_confidence,
            min_margin=self.decisions.min_margin,
            distribution=_tool_distribution(out),
            execute=execute,
        )
        out.update(decision, latency_ms=(perf_counter() - started) * 1000.0)
        return out

    def reset_adaptation(self) -> None:
        """Restore the adapter's initial snapshot."""
        if self.adapter is None:
            raise ValueError("No AdaptiveInference adapter is configured")
        self.adapter.reset()


System1Model = System1


__all__ = [
    "AdaptationTrace",
    "AdaptiveInference",
    "DecisionSpace",
    "GroundedDecision",
    "NeuralMemory",
    "System1",
    "System1Model",
]
