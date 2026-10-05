"""Tests for Reactor: a model-agnostic retrieval memory for System 1 decisions (no TensorFlow needed)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.got import DocumentGraph
from gpbacay_arcane.reactor import Reactor
from gpbacay_arcane.tools import ToolParam, ToolSpec

TICKETS = [("I was charged twice", "billing"), ("refund my last invoice", "billing"),
           ("my parcel never arrived", "shipping"), ("track my delivery", "shipping")]


class StubArc1:
    """Arc1Agent stand-in: classify always prefers the first label; run records the tool prior."""

    def classify(self, text, labels, **kwargs):
        dist = {lab: (0.7 if i == 0 else 0.3 / (len(labels) - 1)) for i, lab in enumerate(labels)}
        return {"label": labels[0], "confidence": 0.7, "distribution": dist, "latency_ms": 1.0}

    def run(self, prompt, tools=None, tool_prior=None, **kwargs):
        return {"function_calls": [], "tool_prior": tool_prior}


def reactor(model=None, graph=None):
    r = Reactor(model, graph)
    for text, label in TICKETS:
        r.remember(text, label)
    return r


def test_memory_overrules_any_model_and_falls_back_to_it():
    first_label = lambda text, labels: {lab: (0.7 if i == 0 else 0.3) for i, lab in enumerate(labels)}  # any callable
    for model in (StubArc1(), first_label):
        out = reactor(model).classify("why was my card charged twice for the invoice", ["shipping", "billing"])
        assert out["label"] == "billing" and out["source"] == "model+memory"
        assert {h["label"] for h in out["neighbors"]} == {"billing"}
        assert abs(sum(out["distribution"].values()) - 1) < 1e-9
        assert reactor(model).classify("where is my parcel", ["sales", "account"])["source"] == "model"
    assert reactor(StubArc1()).classify("track it", ["shipping", "billing"])["latency_ms"] == 1.0  # model fields kept


def test_memory_only_decides_or_abstains():
    r = reactor()
    assert r.classify("track my parcel", ["billing", "shipping"])["label"] == "shipping"
    out = r.classify("hello there", ["billing", "shipping"])
    assert out["label"] is None and out["source"] == "model"  # nothing similar and no model: abstain
    assert r.react("charged twice")["votes"] == {"billing": 1.0}
    with pytest.raises(TypeError):
        r.run("set an alarm", tools=[])


def test_memory_is_dynamic():
    r = reactor()
    r.remember("I was charged twice", "fraud")  # same text again: the label is corrected, not duplicated
    assert len(r.labels()) == 4 and r.classify("charged twice", ["billing", "fraud"])["label"] == "fraud"
    assert r.forget("I was charged twice") and len(r.labels()) == 3
    assert r.react("charged twice")["votes"] == {} and not r.forget("I was charged twice")


def test_tool_prior_counts_votes_for_other_tools_and_no_tool():
    g = DocumentGraph()
    g.add_document("# Guide\n\n## Deploy\nPush the build to the server.", "Guide")  # shared graph: documents never vote
    r = Reactor(StubArc1(), g)
    r.remember("set an alarm for 7am", "set_alarm")
    r.remember("wake me up at six with an alarm", "set_alarm")
    r.remember("thanks, alarm sounds good", None)
    prior = r.run("set an alarm for 6am")["tool_prior"]
    assert set(prior) == {"set_alarm"} and 0.5 < prior["set_alarm"] < 1.0  # the None example took a share
    assert r.run("deploy the build")["tool_prior"] is None  # nothing similar remembered: the model alone

    restored = Reactor(StubArc1(), DocumentGraph.from_json(json.loads(json.dumps(g.to_json()))))
    assert restored.labels() == r.labels()  # the memory persists with the graph
    assert restored.react("set an alarm")["votes"].keys() == {"set_alarm", None}


class BadSpans:
    """A tool caller that fires rate_movie with a wrong title and never calls toggle_bluetooth."""

    def run(self, prompt, tools=None, tool_prior=None, execute=True, **kwargs):
        calls = [{"name": "rate_movie", "arguments": {"title": "away deserves", "stars": 5}}] if "rate" in prompt else []
        return {"function_calls": calls, "confidence": 0.9}


RATE = ToolSpec("rate_movie", "Rate a movie", [ToolParam("title"), ToolParam("stars", "integer")],
                handler=lambda title, stars: f"{title}={stars}")
BLUETOOTH = ToolSpec("toggle_bluetooth", "Turn bluetooth on or off", [ToolParam("enabled", "boolean")])


def test_remembered_requests_fill_arguments_for_any_model():
    r = Reactor(BadSpans())
    r.remember("rate Dune 4 stars", "rate_movie", {"title": "Dune", "stars": 4})
    r.remember("switch bluetooth off", "toggle_bluetooth", {"enabled": False})

    out = r.run("please rate spirited away 5 stars", tools=[RATE, BLUETOOTH])
    assert out["function_calls"] == [{"name": "rate_movie", "arguments": {"title": "spirited away", "stars": 5}}]
    assert out["results"] == ["spirited away=5"] and out["memory_arguments"] == ["rate_movie"]
    assert out["confidence"] is None  # ARC 1's calibrated probability was for the arguments memory replaced

    out = r.run("ok. Also switch bluetooth off", tools=[RATE, BLUETOOTH], execute=False)
    assert out["function_calls"] == [{"name": "toggle_bluetooth", "arguments": {"enabled": False}}]  # held back, added
    assert out["confidence"] is None and out["results"] == []
    assert r.run("switch bluetooth on", tools=[BLUETOOTH])["function_calls"] == []  # "off" is part of the pattern

    restored = Reactor(BadSpans(), DocumentGraph.from_json(json.loads(json.dumps(r.graph.to_json()))))
    assert restored.arguments("rate Coco two stars", RATE) == {"title": "Coco", "stars": 2}
    assert restored.arguments("rate it", RATE) is None


def test_a_model_call_copying_words_a_pattern_explains_is_dropped():
    class WrongTool:  # fires play_podcast with the whole request, as arc1-tiny does on unseen tools
        def run(self, prompt, tools=None, tool_prior=None, execute=True, **kwargs):
            return {"function_calls": [{"name": "play_podcast", "arguments": {"name": prompt}}]}

    podcast = ToolSpec("play_podcast", "Play a podcast", [ToolParam("name")])
    r = Reactor(WrongTool())
    r.remember("Inception deserves 3 stars", "rate_movie", {"title": "Inception", "stars": 3})
    out = r.run("Parasite deserves 5 stars", tools=[RATE, podcast], execute=False)
    assert out["function_calls"] == [{"name": "rate_movie", "arguments": {"title": "Parasite", "stars": 5}}]
    assert r.run("play Serial", tools=[RATE, podcast], execute=False)["function_calls"][0]["name"] == "play_podcast"
