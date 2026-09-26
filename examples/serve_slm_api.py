"""FastAPI server for ARCANE small-language-model chat.

Used by the docs site at /docs/chat.

Model selection, in priority order:
  1. SLM_CONFIG_PATH  -- a saved ArcaneSLMConfig JSON (what the distillation
     run writes next to its weights). Use this for distilled checkpoints, whose
     geometry does not match any named preset.
  2. SLM_PRESET       -- tiny | 100m | distill

Tokenizer, in priority order:
  1. SLM_VOCAB_ADAPTER -- a QwenVocabAdapter JSON (distilled models)
  2. SLM_TOKENIZER_PATH -- a BytePairTokenizer JSON
  3. byte-level fallback with no merges

If nothing is set, the server auto-discovers a distilled checkpoint in Models/
(ARC 1 LM ``arc1_lm.*`` first, then ``arcane_slm_distilled.*``) and falls back
to the tiny preset. A config with ARC 1 keys builds ``Arc1LanguageModel``.

Run from repo root:
  python examples/serve_slm_api.py

Optional env:
  SLM_PRESET=tiny|100m|distill
  SLM_CONFIG_PATH=Models/arcane_slm_distilled.config.json
  SLM_WEIGHTS_PATH=Models/arcane_slm_distilled.weights.h5
  SLM_VOCAB_ADAPTER=Models/qwen_vocab_adapter.json
  SLM_TOKENIZER_PATH=Models/arcane_slm_tiny_tokenizer.json
  PORT=8001
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
# Distilled chat encodes with Qwen's tokenizer via `transformers`. Keep torch
# disabled so AutoTokenizer does not load a NumPy-1.x torch wheel.
os.environ.setdefault("USE_TORCH", "0")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel
from gpbacay_arcane.language_model import ArcaneSLMConfig, ArcaneSmallLanguageModel
from gpbacay_arcane.tokenization import BASE_VOCAB, EOS_ID, BytePairTokenizer

app = FastAPI(title="ARCANE SLM Chat", version="0.2.0")
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


# Auto-discovery: a distilled checkpoint wins over the tiny default, because if
# one exists it is almost certainly what the operator wants served.
DISTILLED_ADAPTER = os.path.join(MODELS_DIR, "qwen_vocab_adapter.json")
# (config, weights) pairs, most preferred first; the first complete pair wins.
_CANDIDATES = [
    (os.path.join(MODELS_DIR, f"{stem}.config.json"), os.path.join(MODELS_DIR, f"{stem}.weights.h5"))
    for stem in ("arc1_lm", "arcane_slm_distilled")
]
DISTILLED_CONFIG, DISTILLED_WEIGHTS = next(
    ((c, w) for c, w in _CANDIDATES if os.path.exists(c) and os.path.exists(w)), ("", "")
)

CONFIG_PATH = _env("SLM_CONFIG_PATH") or DISTILLED_CONFIG
PRESET = (_env("SLM_PRESET") or ("distill" if CONFIG_PATH else "tiny")).lower()

WEIGHTS_PATH = _env("SLM_WEIGHTS_PATH") or _first_existing(
    DISTILLED_WEIGHTS if CONFIG_PATH else "",
    os.path.join(MODELS_DIR, f"arcane_slm_{PRESET}.weights.h5"),
)
VOCAB_ADAPTER_PATH = _env("SLM_VOCAB_ADAPTER") or _first_existing(
    DISTILLED_ADAPTER if CONFIG_PATH else ""
)
TOKENIZER_PATH = _env("SLM_TOKENIZER_PATH") or _first_existing(
    os.path.join(MODELS_DIR, f"arcane_slm_{PRESET}_tokenizer.json")
)

_model: Any = None
_tokenizer: Any = None
_allowed_ids: Optional[List[int]] = None
_state = {
    "ready": False,
    "trained": False,
    "error": None,
    "preset": PRESET,
    "tokenizer": None,
    "vocab_size": None,
    "parameters": None,
    "distilled": False,
}


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list)
    max_new_tokens: int = Field(default=48, ge=1, le=128)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    trained: bool
    preset: str


def _build_prompt(history: List[ChatTurn], message: str) -> str:
    # These checkpoints are next-token LMs, not instruction-tuned. Feed the
    # conversation as a continuation seed so sampling stays in-distribution.
    parts: List[str] = []
    for turn in history[-6:]:
        text = turn.content.strip()
        if text:
            parts.append(text)
    parts.append(message.strip())
    return "\n".join(parts)


def _load_config():
    """Return ``(config, model_class)``."""
    if CONFIG_PATH:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        if "engram_table_size" in payload:
            print(f"[slm] Loaded ARC 1 LM config {CONFIG_PATH}")
            return Arc1Config.from_dict(payload), Arc1LanguageModel
        # Tolerate configs written by a newer/older version of the dataclass.
        known = {f for f in ArcaneSLMConfig.__dataclass_fields__}
        filtered = {k: v for k, v in payload.items() if k in known}
        dropped = set(payload) - set(filtered)
        if dropped:
            print(f"[slm] Ignoring unknown config keys: {sorted(dropped)}")
        print(f"[slm] Loaded config {CONFIG_PATH}")
        return ArcaneSLMConfig(**filtered), ArcaneSmallLanguageModel
    print(f"[slm] Using preset '{PRESET}'")
    return ArcaneSLMConfig.from_preset(PRESET), ArcaneSmallLanguageModel


def _load_tokenizer(vocab_size: int):
    """Return ``(tokenizer, allowed_generation_ids, label)``."""
    if VOCAB_ADAPTER_PATH:
        from gpbacay_arcane.qwen_vocab import QwenVocabAdapter

        adapter = QwenVocabAdapter.load(VOCAB_ADAPTER_PATH)
        adapter.tokenizer  # fail fast at startup, not on the first chat request
        print(f"[slm] Loaded Qwen vocab adapter {VOCAB_ADAPTER_PATH}")
        return adapter, adapter.generation_ids(), "qwen-adapter"
    if TOKENIZER_PATH:
        tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
        print(f"[slm] Loaded tokenizer {TOKENIZER_PATH}")
        return tokenizer, tokenizer.generation_ids(printable_only=True), "byte-pair"
    tokenizer = BytePairTokenizer(vocab_size=max(vocab_size, BASE_VOCAB))
    print("[slm] Using byte-level tokenizer (no merges)")
    return tokenizer, tokenizer.generation_ids(printable_only=True), "byte-level"


def _load_model() -> None:
    global _model, _tokenizer, _allowed_ids
    try:
        config, model_cls = _load_config()
        print(
            f"[slm] Building {model_cls.__name__} "
            f"d_model={config.d_model} layers={config.num_layers} "
            f"vocab={config.vocab_size} seq_len={config.seq_len}"
        )
        model = model_cls(config)
        model.build_model()

        trained = False
        if WEIGHTS_PATH:
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
            model.load_weights(WEIGHTS_PATH)
            trained = True
            print(f"[slm] Loaded weights {WEIGHTS_PATH}")
        else:
            print("[slm] No weights found; serving untrained weights")

        tokenizer, allowed, label = _load_tokenizer(config.vocab_size)

        _model = model
        _tokenizer = tokenizer
        _allowed_ids = allowed
        _state.update(
            ready=True,
            trained=trained,
            error=None,
            tokenizer=label,
            vocab_size=config.vocab_size,
            parameters=int(model.count_params()),
            distilled=bool(VOCAB_ADAPTER_PATH and trained),
            preset=("arc1-lm" if model_cls is Arc1LanguageModel else "distilled") if CONFIG_PATH else PRESET,
        )
        print(f"[slm] Ready ({_state['parameters']:,} params, tokenizer={label})")
    except Exception as exc:
        _state["ready"] = False
        _state["error"] = str(exc)
        print(f"[slm] Failed to load: {exc}")


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["ready"] else ("error" if _state["error"] else "loading"),
        "ready": _state["ready"],
        "trained": _state["trained"],
        "preset": _state["preset"],
        "tokenizer": _state["tokenizer"],
        "vocab_size": _state["vocab_size"],
        "parameters": _state["parameters"],
        "distilled": _state["distilled"],
        "error": _state["error"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model = _model
    tokenizer = _tokenizer
    if not _state["ready"] or model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail=_state["error"] or "ARCANE SLM is still loading"
        )

    prompt = _build_prompt(req.history, req.message)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    # Keep room in the context window for the tokens we are about to generate.
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
            allowed_token_ids=_allowed_ids,
        )
        reply = tokenizer.decode(out_ids[len(prompt_ids):]).strip()
        if not reply:
            reply = "…"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return ChatResponse(reply=reply, trained=_state["trained"], preset=_state["preset"])


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
