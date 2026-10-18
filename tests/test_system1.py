"""ARCANE 5.0 System 1 composition contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpbacay_arcane.system1 import DecisionSpace, System1
from gpbacay_arcane.tools import ToolParam, ToolSpec


def test_decision_space_enforces_schema_confidence_and_margin():
    space = DecisionSpace(["approve", "review", "decline"], min_confidence=0.6, min_margin=0.2)

    accepted = space.ground({"approve": 8, "review": 1, "decline": 1, "invented": 100})
    assert accepted.label == "approve" and not accepted.abstained
    assert accepted.metadata["ignored_labels"] == ["invented"]
    assert set(accepted.distribution) == {"approve", "review", "decline"}

    ambiguous = space.ground({"approve": 0.51, "review": 0.49})
    assert ambiguous.label is None and ambiguous.reason == "low_confidence"
    assert DecisionSpace(["a", "b"], min_confidence=0.4, min_margin=0.2).ground(
        {"a": 0.55, "b": 0.45}
    ).reason == "ambiguous_decision"


def test_system1_composes_model_memory_and_grounding():
    def predictor(text, labels):
        del text
        return {label: (0.8 if label == "shipping" else 0.2) for label in labels}

    system = System1(
        predictor,
        DecisionSpace(["shipping", "billing"], min_confidence=0.55),
        memory_weight=0.8,
    )
    system.remember("my card was charged twice", "billing")
    decision = system.decide("why did you charge my card twice")

    assert decision.label == "billing"
    assert decision.source == "model+memory"
    assert decision.evidence[0]["label"] == "billing"
    assert decision.latency_ms >= 0.0
    assert decision.to_dict()["label"] == "billing"


def test_system1_can_abstain_without_a_model_or_matching_memory():
    system = System1(None, DecisionSpace(["safe", "unsafe"], min_confidence=0.6))
    decision = system.decide("an unseen request")

    assert decision.abstained and decision.label is None
    assert decision.reason == "no_supported_decision"
    assert decision.source == "memory"


def test_system1_accepts_a_predict_style_model():
    class Predictor:
        def predict(self, inputs, **kwargs):
            assert inputs == [[1, 2]] and kwargs == {"verbose": 0}
            return np.array([[0.75, 0.25]], dtype=np.float32)

        def __call__(self, *args):
            raise AssertionError("predict() should be preferred for model objects")

    decision = System1(Predictor(), DecisionSpace(["yes", "no"])).decide(
        [[1, 2]], model_kwargs={"verbose": 0}
    )
    assert decision.label == "yes"


def test_system1_uses_adapter_predictions_and_reports_trace():
    result = SimpleNamespace(
        predictions=np.array([[0.1, 0.9]], dtype=np.float32),
        baseline_objective=0.8,
        final_objective=0.4,
        best_objective=0.4,
        attempted_steps=2,
        accepted_steps=2,
        rolled_back=False,
        update_norms=(0.001, 0.001),
    )

    class Adapter:
        def adapt(self, inputs, **kwargs):
            assert inputs == "encoded" and kwargs == {"steps": 2}
            return result

        def reset(self):
            pass

    system = System1(
        None,
        DecisionSpace(["deny", "allow"], min_confidence=0.5),
        adapter=Adapter(),
    )
    decision = system.decide(
        "raw input",
        adapt=True,
        adaptation_inputs="encoded",
        adaptation_kwargs={"steps": 2},
    )

    assert decision.label == "allow" and decision.source == "adaptive-model"
    assert decision.adaptation.accepted_steps == 2
    assert decision.adaptation.final_objective == pytest.approx(0.4)


def test_system1_validates_configuration_and_memory_labels():
    with pytest.raises(ValueError):
        DecisionSpace([])
    with pytest.raises(ValueError):
        DecisionSpace(["same", "same"])
    with pytest.raises(ValueError):
        System1(None, DecisionSpace(["a"]), memory_weight=1.1)

    system = System1(None, DecisionSpace(["known"]))
    with pytest.raises(ValueError):
        system.remember("example", "unknown")
    with pytest.raises(ValueError):
        system.decide("example", adapt=True)


def test_system1_holds_a_tool_call_until_it_clears_the_decision_bar():
    ping = ToolSpec("ping", "Ping a host", [ToolParam("host")], handler=lambda host: f"pong {host}")
    fired = [{"name": "ping", "arguments": {"host": "local"}}]
    space = DecisionSpace(["unused"], min_confidence=0.6, min_margin=0.2)

    class Router:
        def __init__(self, confidence, calls, tools=None):
            self.confidence, self.calls, self.tools = confidence, calls, tools

        def run(self, prompt, tools=None, tool_prior=None, execute=True, **kwargs):
            assert execute is False and prompt
            out = {"function_calls": self.calls, "confidence": self.confidence, "source": "model"}
            if self.tools is not None:
                out["decisions"] = {"tools": self.tools}
            return out

    ran = System1(Router(0.9, fired), space).act("ping local", [ping])
    assert ran["results"] == ["pong local"] and not ran["abstained"] and ran["reason"] is None

    held = System1(Router(0.4, fired), space).act("ping local", [ping])
    assert held["function_calls"] == fired and held["results"] == []
    assert held["abstained"] and held["reason"] == "low_confidence"

    vague = System1(Router(None, fired), space).act("ping local", [ping])
    assert vague["reason"] == "uncalibrated" and vague["results"] == []

    close = System1(Router(0.9, fired, {"ping": 0.55, "trace": 0.45}), space).act("ping", [ping])
    assert close["reason"] == "ambiguous_decision" and close["results"] == []

    refusal = System1(Router(0.95, []), space).act("never mind", [ping])
    assert refusal["results"] == [] and not refusal["abstained"] and refusal["reason"] is None

    unseen = System1(None, space).act("hello", [ping])
    assert unseen["abstained"] and unseen["reason"] == "no_supported_decision"
    assert unseen["source"] == "memory"

    preview = System1(Router(0.9, fired), space).act("ping local", [ping], execute=False)
    assert preview["results"] == [] and not preview["abstained"]

    with pytest.raises(TypeError):
        System1(lambda text, labels: {}, space).act("x", [ping])


def test_system1_holds_arguments_taken_from_memory():
    rate = ToolSpec(
        "rate_movie", "Rate a movie",
        [ToolParam("title"), ToolParam("stars", "integer")],
        handler=lambda title, stars: f"{title}={stars}",
    )

    class Confident:
        def run(self, prompt, tools=None, tool_prior=None, execute=True, **kwargs):
            return {
                "function_calls": [{"name": "rate_movie", "arguments": {"title": "wrong", "stars": 1}}],
                "confidence": 0.95,
                "source": "model",
            }

    system = System1(Confident(), DecisionSpace(["unused"], min_confidence=0.5))
    system.memory.remember("rate Dune 4 stars", "rate_movie", {"title": "Dune", "stars": 4})
    out = system.act("please rate spirited away 5 stars", [rate])

    assert out["function_calls"] == [{"name": "rate_movie", "arguments": {"title": "spirited away", "stars": 5}}]
    assert out["reason"] == "uncalibrated" and out["results"] == []
    assert out["source"] == "model+memory" and out["neighbors"]
    assert system.memory.model is None
