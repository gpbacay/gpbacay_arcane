"""Tests for Hippocampus: a model-agnostic retrieval memory for System 1 decisions (no TensorFlow needed)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.got import DocumentGraph
from gpbacay_arcane.hippocampus import Hippocampus
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


def hippocampus(model=None, graph=None):
    r = Hippocampus(model, graph)
    for text, label in TICKETS:
        r.remember(text, label)
    return r


def test_memory_overrules_any_model_and_falls_back_to_it():
    first_label = lambda text, labels: {lab: (0.7 if i == 0 else 0.3) for i, lab in enumerate(labels)}  # any callable
    for model in (StubArc1(), first_label):
        out = hippocampus(model).classify("why was my card charged twice for the invoice", ["shipping", "billing"])
        assert out["label"] == "billing" and out["source"] == "model+memory"
        assert {h["label"] for h in out["neighbors"]} == {"billing"}
        assert abs(sum(out["distribution"].values()) - 1) < 1e-9
        assert hippocampus(model).classify("where is my parcel", ["sales", "account"])["source"] == "model"
    assert hippocampus(StubArc1()).classify("track it", ["shipping", "billing"])["latency_ms"] == 1.0  # model fields kept


def test_memory_only_decides_or_abstains():
    r = hippocampus()
    assert r.classify("track my parcel", ["billing", "shipping"])["label"] == "shipping"
    out = r.classify("hello there", ["billing", "shipping"])
    assert out["label"] is None and out["source"] == "model"  # nothing similar and no model: abstain
    assert r.react("charged twice")["votes"] == {"billing": 1.0}
    with pytest.raises(TypeError):
        r.run("set an alarm", tools=[])


def test_memory_is_dynamic():
    r = hippocampus()
    r.remember("I was charged twice", "fraud")  # same text again: the label is corrected, not duplicated
    assert len(r.labels()) == 4 and r.classify("charged twice", ["billing", "fraud"])["label"] == "fraud"
    assert r.forget("I was charged twice") and len(r.labels()) == 3
    assert r.react("charged twice")["votes"] == {} and not r.forget("I was charged twice")


class Teacher:
    """A stronger model stand-in: knows two topics, counts how often it is asked."""

    def __init__(self, answers=None, fail=False):
        self.calls, self.answers, self.fail = 0, answers or {}, fail

    def __call__(self, text, labels):
        self.calls += 1
        if self.fail:
            raise ConnectionError("teacher offline")
        return next((lab for word, lab in self.answers.items() if word in text), None)


def test_teacher_answers_when_unsure_and_the_memory_keeps_the_answer():
    teacher = Teacher({"charged": "billing", "parcel": "shipping"})
    r = Hippocampus(teacher=teacher)  # no model, no memory: a cold start
    out = r.classify("I was charged twice", ["billing", "shipping"])
    assert (out["label"], out["source"], out["confidence"], teacher.calls) == ("billing", "teacher", 1.0, 1)
    assert list(r.labels().values()) == ["billing"] and r.graph.docs[next(iter(r.labels()))]["source"] == "auto"
    again = r.classify("why was I charged twice this month", ["billing", "shipping"])
    assert (again["label"], again["source"], teacher.calls) == ("billing", "memory", 1)  # decided here, teacher not asked
    assert r.classify("my parcel never arrived", ["billing", "shipping"])["source"] == "teacher" and teacher.calls == 2


def test_teacher_is_not_asked_when_confident_and_is_trusted_only_when_confident():
    teacher = Teacher({"charged": "billing"})
    r = Hippocampus(StubArc1(), teacher=teacher)  # the stub gives its first label 0.7: under confident=0.8
    assert r.classify("I was charged twice", ["billing", "shipping"])["source"] == "teacher" and teacher.calls == 1
    r2 = Hippocampus(lambda text, labels: {lab: (0.95 if i == 0 else 0.05) for i, lab in enumerate(labels)}, teacher=teacher)
    assert r2.classify("I was charged twice", ["billing", "shipping"])["source"] == "model" and teacher.calls == 1
    unsure = Hippocampus(teacher=lambda text, labels: {"billing": 0.6, "shipping": 0.4})
    out = unsure.classify("hello", ["billing", "shipping"])
    assert out["label"] is None and unsure.labels() == {}  # 0.6 < confident: not a source of truth
    assert Hippocampus(teacher=lambda text, labels: "refunds").classify("hello", ["billing"])["label"] is None  # not an offered label


def test_teacher_is_not_asked_once_the_memory_is_decisive():
    teacher = Teacher({"charged": "billing"})
    r = Hippocampus(StubArc1(), teacher=teacher)  # the model always leans to the first label, "shipping"
    options = ["shipping", "billing"]
    for text in ("I was charged twice", "charged for a refund I never asked", "card charged again yesterday"):
        assert r.classify(text, options)["source"] == "teacher"  # the memory's vote (billing) lifts it only to 0.65
    assert teacher.calls == 3 and len(r.labels()) == 3
    out = r.classify("why am I charged every month", options)
    assert (out["label"], out["source"], teacher.calls) == ("billing", "model+memory", 3)  # 3 neighbors back billing


def test_a_failing_teacher_leaves_the_memory_working_and_remembered_texts_are_not_put_to_it():
    r = Hippocampus(teacher=Teacher(fail=True))
    out = r.classify("hello", ["billing", "shipping"])
    assert out["label"] is None and "ConnectionError: teacher offline" in out["teacher_error"] and r.labels() == {}
    teacher = Teacher({"charged": "billing"})
    r = Hippocampus(teacher=teacher)
    r.remember("I was charged twice", "fraud")
    assert r.classify("I was charged twice", ["billing", "fraud"])["label"] == "fraud" and teacher.calls == 0
    capped = Hippocampus(teacher=Teacher({"a": "x", "b": "y"}), max_size=1)
    capped.classify("a one", ["x", "y"])
    capped.classify("b two", ["x", "y"])
    assert list(capped.labels().values()) == ["y"]  # the cap applies to what the teacher taught


def test_observe_stores_only_what_is_surprising():
    r = Hippocampus()
    assert r.observe("I was charged twice", "billing")  # nothing known: stored
    assert r.observe("  I WAS charged   twice ", "billing") is None  # same text, already decided: skipped
    assert r.observe("track my delivery", "shipping") and len(r.labels()) == 2
    assert r.observe("I was charged twice", "fraud")  # the outcome disagrees: corrected, not duplicated
    assert len(r.labels()) == 2 and r.classify("charged twice", ["billing", "fraud"])["label"] == "fraud"
    assert Hippocampus(StubArc1()).observe("my parcel never arrived", "shipping")  # model unsure (0.7) and no memory


def test_observe_judges_over_the_given_decision_space_and_spares_remembered_examples():
    def build():
        r = Hippocampus()
        r.remember("alpha beta", "x")
        r.remember("alpha beta gamma delta epsilon", "o")
        r.remember("alpha beta gamma zeta", "o")
        return r

    assert build().observe("alpha beta gamma", "x")  # among x and o, memory leans o: surprising
    assert build().observe("alpha beta gamma", "x", labels=["x", "z"]) is None  # o is another decision: x is decided
    r = build()
    assert r.observe("alpha beta", "z") is None and r.react("alpha beta", ["x"])["votes"] == {"x": 1.0}  # remembered: untouched
    r.forget("alpha beta")
    assert r.observe("alpha beta", "z")  # forgotten first, then observe may write


def test_observe_needs_repeated_trusted_evidence_to_commit():
    r = Hippocampus(threshold=1.0, leak=0.5)
    assert r.observe("refund my invoice", "billing", trust=0.6) is None  # 0.6
    assert r.observe("refund my invoice", "shipping", trust=0.6) is None  # billing decays to 0.3, then is cancelled
    assert r.observe("refund my invoice", "billing", trust=0.6) is None  # shipping cancelled, billing starts over
    assert r.observe("refund my invoice", "billing", trust=0.6) is None  # 0.3 + 0.6 = 0.9 < 1.0
    assert r.labels() == {}
    assert r.observe("refund my invoice", "billing", trust=0.6)  # 0.45 + 0.6 >= 1.0: committed
    assert list(r.labels().values()) == ["billing"]


def test_pending_evidence_does_not_decay_by_default_and_is_bounded():
    r = Hippocampus(threshold=1.0, max_pending=2)
    assert r.observe("refund my invoice", "billing", trust=0.6) is None
    for i in range(50):
        r.observe(f"unrelated filler number {i} zzz{i}", "other", trust=0.1)  # no decay, so these never push it out...
    assert r.observe("refund my invoice", "billing", trust=0.6) is None  # ...but max_pending=2 did: starts over at 0.6
    assert r.observe("refund my invoice", "billing", trust=0.6)  # 0.6 + 0.6: committed
    r = Hippocampus(threshold=1.0)
    r.observe("refund my invoice", "billing", trust=0.6)
    for i in range(50):
        r.observe(f"unrelated filler number {i} zzz{i}", "other", trust=0.1)
    assert r.observe("refund my invoice", "billing", trust=0.6)  # leak=0: the old evidence still counts


def test_eviction_spares_manual_and_drops_the_oldest_observed():
    r = Hippocampus(max_size=3)
    r.remember("pinned example one", "a")
    r.observe("alpha beta gamma", "a")
    r.observe("delta epsilon zeta", "b")
    r.observe("eta theta iota", "c")  # over the cap: the oldest unused auto example goes
    assert len(r.labels()) == 3 and r.react("alpha beta gamma")["votes"] == {}
    assert r.react("delta epsilon zeta")["votes"] == {"b": 1.0}
    assert "a" in r.labels().values() and r.react("pinned example one")["votes"] == {"a": 1.0}
    manual = Hippocampus(max_size=1)
    manual.remember("one", "a"), manual.remember("two", "b")
    assert len(manual.labels()) == 2  # remember is never capped


def test_observe_state_persists_with_the_graph():
    r = Hippocampus(max_size=5)
    r.observe("I was charged twice", "billing")
    restored = Hippocampus(graph=DocumentGraph.from_json(json.loads(json.dumps(r.graph.to_json()))))
    assert restored.graph.docs[next(iter(restored.labels()))]["source"] == "auto"


def test_tool_prior_counts_votes_for_other_tools_and_no_tool():
    g = DocumentGraph()
    g.add_document("# Guide\n\n## Deploy\nPush the build to the server.", "Guide")  # shared graph: documents never vote
    r = Hippocampus(StubArc1(), g)
    r.remember("set an alarm for 7am", "set_alarm")
    r.remember("wake me up at six with an alarm", "set_alarm")
    r.remember("thanks, alarm sounds good", None)
    prior = r.run("set an alarm for 6am")["tool_prior"]
    assert set(prior) == {"set_alarm"} and 0.5 < prior["set_alarm"] < 1.0  # the None example took a share
    assert r.run("deploy the build")["tool_prior"] is None  # nothing similar remembered: the model alone

    restored = Hippocampus(StubArc1(), DocumentGraph.from_json(json.loads(json.dumps(g.to_json()))))
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
    r = Hippocampus(BadSpans())
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

    restored = Hippocampus(BadSpans(), DocumentGraph.from_json(json.loads(json.dumps(r.graph.to_json()))))
    assert restored.arguments("rate Coco two stars", RATE) == {"title": "Coco", "stars": 2}
    assert restored.arguments("rate it", RATE) is None


def test_a_model_call_copying_words_a_pattern_explains_is_dropped():
    class WrongTool:  # fires play_podcast with the whole request, as arc1-tiny does on unseen tools
        def run(self, prompt, tools=None, tool_prior=None, execute=True, **kwargs):
            return {"function_calls": [{"name": "play_podcast", "arguments": {"name": prompt}}]}

    podcast = ToolSpec("play_podcast", "Play a podcast", [ToolParam("name")])
    r = Hippocampus(WrongTool())
    r.remember("Inception deserves 3 stars", "rate_movie", {"title": "Inception", "stars": 3})
    out = r.run("Parasite deserves 5 stars", tools=[RATE, podcast], execute=False)
    assert out["function_calls"] == [{"name": "rate_movie", "arguments": {"title": "Parasite", "stars": 5}}]
    assert r.run("play Serial", tools=[RATE, podcast], execute=False)["function_calls"][0]["name"] == "play_podcast"
