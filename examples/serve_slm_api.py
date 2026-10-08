"""FastAPI server for the ARCANE documentation chat.

The server exposes every locally available ARC 1 language-model checkpoint and
lets each chat request choose a model.  With the repository defaults this is:

* ``arc1-lm-100m`` -- the compact 16-layer model, when trained weights exist
* ``arc1-lm-v2`` -- the 12-layer hybrid convolution/GQA model (default fallback)
* ``arc1-lm-v1`` -- the original six-layer causal ARC 1 model

An explicit ``SLM_CONFIG_PATH`` keeps the old single-model deployment mode.

Run from the repository root:

    python examples/serve_slm_api.py

Optional environment variables:

    SLM_DEFAULT_MODEL=arc1-lm-100m|arc1-lm-v2|arc1-lm-v1
    SLM_CONFIG_PATH=Models/custom.config.json
    SLM_WEIGHTS_PATH=Models/custom.weights.h5
    SLM_MODEL_ID=custom
    SLM_PRESET=tiny|100m|distill
    SLM_VOCAB_ADAPTER=Models/qwen_vocab_adapter.json
    SLM_TOKENIZER_PATH=Models/arcane_slm_tiny_tokenizer.json
    PORT=8001
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("USE_TORCH", "0")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel
from gpbacay_arcane.language_model import ArcaneSLMConfig, ArcaneSmallLanguageModel
from gpbacay_arcane.tokenization import BASE_VOCAB, EOS_ID, BytePairTokenizer

app = FastAPI(title="ARCANE SLM Chat", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(ROOT_DIR, "Models")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _first_existing(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return ""


def _arc1_spec(model_id: str, stem: str, label: str, quality_note: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "label": label,
        "config_path": os.path.join(MODELS_DIR, f"{stem}.config.json"),
        "weights_path": os.path.join(MODELS_DIR, f"{stem}.weights.h5"),
        "preset": model_id,
        "quality_note": quality_note,
    }


def _discover_specs() -> Dict[str, Dict[str, Any]]:
    explicit_config = _env("SLM_CONFIG_PATH")
    if explicit_config:
        model_id = _env("SLM_MODEL_ID") or "configured"
        inferred_weights = (
            explicit_config[: -len(".config.json")] + ".weights.h5"
            if explicit_config.endswith(".config.json")
            else ""
        )
        return {
            model_id: {
                "id": model_id,
                "label": "Configured ARCANE model",
                "config_path": explicit_config,
                "weights_path": _env("SLM_WEIGHTS_PATH") or _first_existing(inferred_weights),
                "preset": _env("SLM_PRESET") or "distill",
                "quality_note": "Operator-configured checkpoint.",
            }
        }

    candidates = [
        _arc1_spec(
            "arc1-lm-100m",
            "arc1_lm_100m",
            "ARC 1 LM 100M · compact chat",
            "Compact hybrid model distilled with assistant-only supervision.",
        ),
        _arc1_spec(
            "arc1-lm-v2",
            "arc1_lm_v2",
            "ARC 1 LM v2 · hybrid",
            "Experimental checkpoint with only bounded smoke training; integration-ready, not quality-equivalent to LFM2.5.",
        ),
        _arc1_spec(
            "arc1-lm-v1",
            "arc1_lm",
            "ARC 1 LM v1 · legacy",
            "Older TinyStories checkpoint; stronger held-out loss than the current v2 smoke checkpoint.",
        ),
    ]
    # A config is useful for training, but listing a model in chat before its
    # checkpoint exists creates a selectable option that can never load.
    specs = {
        spec["id"]: spec
        for spec in candidates
        if os.path.exists(spec["config_path"]) and os.path.exists(spec["weights_path"])
    }
    if specs:
        return specs

    preset = (_env("SLM_PRESET") or "tiny").lower()
    return {
        preset: {
            "id": preset,
            "label": f"ARCANE SLM · {preset}",
            "config_path": "",
            "weights_path": _first_existing(
                _env("SLM_WEIGHTS_PATH"),
                os.path.join(MODELS_DIR, f"arcane_slm_{preset}.weights.h5"),
            ),
            "preset": preset,
            "quality_note": "Preset model.",
        }
    }


MODEL_SPECS = _discover_specs()
DEFAULT_MODEL_ID = _env("SLM_DEFAULT_MODEL") or (
    "arc1-lm-100m" if "arc1-lm-100m" in MODEL_SPECS else
    "arc1-lm-v2" if "arc1-lm-v2" in MODEL_SPECS else next(iter(MODEL_SPECS))
)
if DEFAULT_MODEL_ID not in MODEL_SPECS:
    print(f"[slm] Unknown SLM_DEFAULT_MODEL={DEFAULT_MODEL_ID!r}; using the first available model")
    DEFAULT_MODEL_ID = next(iter(MODEL_SPECS))

VOCAB_ADAPTER_PATH = _env("SLM_VOCAB_ADAPTER") or _first_existing(
    os.path.join(MODELS_DIR, "qwen_vocab_adapter.json")
)
TOKENIZER_PATH = _env("SLM_TOKENIZER_PATH")

_models: Dict[str, Dict[str, Any]] = {}


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list)
    model: Optional[str] = None
    max_new_tokens: int = Field(default=48, ge=1, le=128)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    trained: bool
    preset: str
    model: str
    parameters: int


def _build_prompt(history: List[ChatTurn], message: str) -> str:
    # These checkpoints are continuation models, not instruction-tuned chat
    # models. Keep the prompt close to their language-model objective.
    parts: List[str] = []
    for turn in history[-6:]:
        text = turn.content.strip()
        if text:
            parts.append(text)
    parts.append(message.strip())
    return "\n".join(parts) + "\n"


def _trim_to_sentence(text: str) -> str:
    """Drop a trailing half-sentence cut off by max_new_tokens."""
    end = max(text.rfind(c) for c in '.!?"')
    return text[: end + 1] if end > 0 else text


def _load_config(spec: Dict[str, Any]):
    """Return ``(config, model_class, displayed_preset)`` for one registry item."""
    config_path = spec["config_path"]
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            payload = json.load(f)
        if "engram_table_size" in payload:
            return Arc1Config.from_dict(payload), Arc1LanguageModel, spec["preset"]
        known = set(ArcaneSLMConfig.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in known}
        return ArcaneSLMConfig(**filtered), ArcaneSmallLanguageModel, spec["preset"]
    preset = spec["preset"]
    return ArcaneSLMConfig.from_preset(preset), ArcaneSmallLanguageModel, preset


def _load_tokenizer(vocab_size: int):
    """Return ``(tokenizer, allowed_generation_ids, label)``."""
    if VOCAB_ADAPTER_PATH:
        from gpbacay_arcane.qwen_vocab import QwenVocabAdapter

        adapter = QwenVocabAdapter.load(VOCAB_ADAPTER_PATH)
        adapter.tokenizer  # Fail at startup rather than the first chat request.
        if adapter.vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary {adapter.vocab_size} does not match model vocabulary {vocab_size}"
            )
        return adapter, adapter.generation_ids(), "qwen-adapter"
    if TOKENIZER_PATH:
        tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
        return tokenizer, tokenizer.generation_ids(printable_only=True), "byte-pair"
    tokenizer = BytePairTokenizer(vocab_size=max(vocab_size, BASE_VOCAB))
    return tokenizer, tokenizer.generation_ids(printable_only=True), "byte-level"


def _public_model(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in {"model", "tokenizer", "allowed_ids"}}


def _load_model(model_id: str, spec: Dict[str, Any]) -> None:
    entry: Dict[str, Any] = {
        "id": model_id,
        "label": spec["label"],
        "ready": False,
        "trained": False,
        "status": "loading",
        "error": None,
        "preset": spec["preset"],
        "tokenizer_name": None,
        "vocab_size": None,
        "parameters": None,
        "context": None,
        "architecture": None,
        "quality_note": spec["quality_note"],
    }
    _models[model_id] = entry
    try:
        config, model_cls, displayed_preset = _load_config(spec)
        print(
            f"[slm] Building {model_id}: {model_cls.__name__} "
            f"d_model={config.d_model} layers={config.num_layers} vocab={config.vocab_size}"
        )
        model = model_cls(config)
        model.build_model()

        weights_path = spec["weights_path"]
        trained = bool(weights_path and os.path.exists(weights_path))
        if weights_path and not trained:
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if trained:
            model.load_weights(weights_path)
            print(f"[slm] Loaded {model_id} weights from {weights_path}")

        tokenizer, allowed_ids, tokenizer_name = _load_tokenizer(config.vocab_size)
        architecture = getattr(config, "lm_architecture", "legacy")
        entry.update(
            model=model,
            tokenizer=tokenizer,
            allowed_ids=allowed_ids,
            ready=True,
            trained=trained,
            status="ok",
            preset=displayed_preset,
            tokenizer_name=tokenizer_name,
            vocab_size=config.vocab_size,
            parameters=int(model.count_params()),
            context=config.seq_len,
            architecture=architecture,
        )
        print(f"[slm] Ready: {model_id} ({entry['parameters']:,} parameters)")
    except Exception as exc:
        entry.update(status="error", error=str(exc))
        print(f"[slm] Failed to load {model_id}: {exc}")


def _load_models() -> None:
    for model_id, spec in MODEL_SPECS.items():
        _load_model(model_id, spec)


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get("/health")
def health():
    default = _models.get(DEFAULT_MODEL_ID)
    models = [_public_model(_models[mid]) for mid in MODEL_SPECS if mid in _models]
    ready = bool(default and default["ready"])
    return {
        # Keep the legacy top-level fields for older clients.
        "status": "ok" if ready else (default["status"] if default else "loading"),
        "ready": ready,
        "trained": bool(default and default["trained"]),
        "preset": default["preset"] if default else DEFAULT_MODEL_ID,
        "tokenizer": default["tokenizer_name"] if default else None,
        "vocab_size": default["vocab_size"] if default else None,
        "parameters": default["parameters"] if default else None,
        "distilled": bool(default and default["trained"] and default["tokenizer_name"] == "qwen-adapter"),
        "error": default["error"] if default else None,
        "default_model": DEFAULT_MODEL_ID,
        "models": models,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model_id = req.model or DEFAULT_MODEL_ID
    entry = _models.get(model_id)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model_id!r}. Available: {', '.join(MODEL_SPECS)}",
        )
    if not entry["ready"]:
        raise HTTPException(status_code=503, detail=entry["error"] or f"{model_id} is still loading")

    model = entry["model"]
    tokenizer = entry["tokenizer"]
    prompt = _build_prompt(req.history, req.message)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    budget = model.slm_config.seq_len - req.max_new_tokens
    if budget > 0:
        prompt_ids = prompt_ids[-budget:]
    try:
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=20,
            eos_id=EOS_ID,
            allowed_token_ids=entry["allowed_ids"],
        )
        reply = _trim_to_sentence(tokenizer.decode(out_ids[len(prompt_ids):]).strip())
        if not reply:
            reply = "…"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return ChatResponse(
        reply=reply,
        trained=entry["trained"],
        preset=entry["preset"],
        model=model_id,
        parameters=entry["parameters"],
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
