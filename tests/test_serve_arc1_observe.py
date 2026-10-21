"""The ARC 1 demo server's /observe: a thumbs up says whether a confirmed decision is worth remembering."""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


class StubArc1:
    """Arc1Agent stand-in: unsure of everything (0.6 for the first label), and records the kwargs it was given."""

    def __init__(self):
        self.kwargs = None

    def classify(self, text, labels, **kwargs):
        self.kwargs = kwargs
        return {"label": labels[0], "confidence": 0.6,
                "distribution": {lab: (0.6 if i == 0 else 0.4 / (len(labels) - 1)) for i, lab in enumerate(labels)}}


@pytest.fixture()
def server():
    spec = importlib.util.spec_from_file_location("serve_arc1_api", os.path.join(ROOT, "examples", "serve_arc1_api.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pydantic resolves the server's string annotations through it
    spec.loader.exec_module(module)
    module._agent, module._model = StubArc1(), object()
    module._state["ready"] = True
    return module, TestClient(module.app)


def test_observe_stores_only_what_the_memory_and_model_are_unsure_of(server):
    module, client = server
    assert client.get("/health").json()["observe"] is True
    body = {"text": "I was charged twice", "label": "billing", "labels": ["billing", "shipping"], "memory": []}
    assert client.post("/observe", json=body).json() == {"stored": True}  # model gives billing only 0.6: not confident

    known = {**body, "memory": [{"text": "I was charged twice", "label": "billing"}]}
    assert client.post("/observe", json=known).json() == {"stored": False}  # taught by hand already: left alone

    similar = {**body, "text": "charged twice this month",
               "memory": [{"text": "I was charged twice", "label": "billing"}]}
    assert client.post("/observe", json=similar).json() == {"stored": False}  # memory (1.0) and model (0.6) agree: 0.8


def test_observe_passes_the_scene_task_to_the_model_and_keeps_tool_arguments(server):
    module, client = server
    body = {"text": "rate Dune 4 stars", "label": "rate_movie", "arguments": {"title": "Dune", "stars": 4},
            "labels": ["rate_movie", "get_weather"], "task": "Pick the tool.", "descriptions": {"get_weather": "Weather"}}
    assert client.post("/observe", json=body).json() == {"stored": True}
    assert module._agent.kwargs == {"task": "Pick the tool.", "descriptions": {"get_weather": "Weather"}}

    no_tool = {"text": "thanks, that is all", "label": None, "labels": ["rate_movie", "get_weather"]}
    assert client.post("/observe", json=no_tool).json() == {"stored": True}  # nothing similar remembered yet


def test_observe_waits_for_the_model_and_limits_its_input(server):
    module, client = server
    assert client.post("/observe", json={"text": "x" * 501, "label": "a"}).status_code == 422
    module._state["ready"] = False
    assert client.post("/observe", json={"text": "hello", "label": "a"}).status_code == 503
