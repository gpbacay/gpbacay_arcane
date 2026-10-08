"""Selection and metadata tests for the documentation chat API."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_SERVER_PATH = Path(__file__).parents[1] / "examples" / "serve_slm_api.py"
_SPEC = importlib.util.spec_from_file_location("arcane_serve_slm_api", _SERVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


class _Tokenizer:
    def encode(self, text, add_bos=False):
        assert text.endswith("\n")
        return [2, 10] if add_bos else [10]

    def decode(self, token_ids):
        return "A selected-model answer."


class _Model:
    slm_config = SimpleNamespace(seq_len=64)

    def generate(self, token_ids, **kwargs):
        assert kwargs["max_new_tokens"] == 8
        return list(token_ids) + [42]


@pytest.fixture
def fake_registry(monkeypatch):
    entry = {
        "id": "arc1-lm-v2",
        "label": "ARC 1 LM v2",
        "ready": True,
        "trained": True,
        "status": "ok",
        "error": None,
        "preset": "arc1-lm-v2",
        "tokenizer_name": "test",
        "vocab_size": 100,
        "parameters": 12,
        "context": 64,
        "architecture": "hybrid",
        "quality_note": "test",
        "model": _Model(),
        "tokenizer": _Tokenizer(),
        "allowed_ids": [42],
    }
    monkeypatch.setattr(server, "DEFAULT_MODEL_ID", "arc1-lm-v2")
    monkeypatch.setattr(server, "MODEL_SPECS", {"arc1-lm-v2": {}})
    monkeypatch.setattr(server, "_models", {"arc1-lm-v2": entry})
    return entry


def test_health_lists_default_selectable_model_without_runtime_objects(fake_registry):
    result = server.health()
    assert result["default_model"] == "arc1-lm-v2"
    assert result["models"][0]["architecture"] == "hybrid"
    assert "model" not in result["models"][0]
    assert "tokenizer" not in result["models"][0]


def test_chat_uses_requested_model_and_reports_it(fake_registry):
    response = server.chat(
        server.ChatRequest(
            message="hello",
            model="arc1-lm-v2",
            max_new_tokens=8,
            temperature=0.0,
        )
    )
    assert response.model == "arc1-lm-v2"
    assert response.reply == "A selected-model answer."
    assert response.trained is True


def test_chat_rejects_unknown_model(fake_registry):
    with pytest.raises(HTTPException) as caught:
        server.chat(server.ChatRequest(message="hello", model="missing"))
    assert caught.value.status_code == 400
    assert "Available: arc1-lm-v2" in caught.value.detail
