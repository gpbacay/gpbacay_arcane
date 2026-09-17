"""FastAPI server for ARCANE small-language-model chat.

Used by the docs site at /docs/chat. Defaults to the tiny preset so a
CPU machine can answer; set SLM_PRESET=100m for the full decoder.

Run from repo root:
  python examples/serve_slm_api.py

Optional env:
  SLM_PRESET=tiny|100m
  SLM_WEIGHTS_PATH=path/to/weights.h5
  SLM_TOKENIZER_PATH=path/to/tokenizer.json
  PORT=8001
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gpbacay_arcane.language_model import ArcaneSmallLanguageModel
from gpbacay_arcane.tokenization import BASE_VOCAB, EOS_ID, BytePairTokenizer

app = FastAPI(title="ARCANE SLM Chat", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRESET = os.environ.get("SLM_PRESET", "tiny").strip().lower()
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "Models", f"arcane_slm_{PRESET}.weights.h5")
DEFAULT_TOKENIZER = os.path.join(ROOT_DIR, "Models", f"arcane_slm_{PRESET}_tokenizer.json")
WEIGHTS_PATH = os.environ.get("SLM_WEIGHTS_PATH", "").strip() or (
    DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else ""
)
TOKENIZER_PATH = os.environ.get("SLM_TOKENIZER_PATH", "").strip() or (
    DEFAULT_TOKENIZER if os.path.exists(DEFAULT_TOKENIZER) else ""
)

_model: Optional[ArcaneSmallLanguageModel] = None
_tokenizer: Optional[BytePairTokenizer] = None
_state = {
    "ready": False,
    "trained": False,
    "error": None,
    "preset": PRESET,
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
    # Tiny/100m checkpoints are next-token LMs, not chat-tuned. Feed the
    # user's text as a continuation seed so sampling stays in-distribution.
    parts: List[str] = []
    for turn in history[-6:]:
        text = turn.content.strip()
        if text:
            parts.append(text)
    parts.append(message.strip())
    return "\n".join(parts)


def _load_model() -> None:
    global _model, _tokenizer
    try:
        print(f"[slm] Building ArcaneSmallLanguageModel preset={PRESET}")
        model = ArcaneSmallLanguageModel.from_preset(PRESET)
        model.build_model()
        trained = False
        if WEIGHTS_PATH:
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
            model.load_weights(WEIGHTS_PATH)
            trained = True
            print(f"[slm] Loaded weights {WEIGHTS_PATH}")
        else:
            print("[slm] No SLM_WEIGHTS_PATH set; serving untrained weights")

        if TOKENIZER_PATH and os.path.exists(TOKENIZER_PATH):
            tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
            print(f"[slm] Loaded tokenizer {TOKENIZER_PATH}")
        else:
            tokenizer = BytePairTokenizer(vocab_size=max(model.slm_config.vocab_size, BASE_VOCAB))
            print("[slm] Using byte-level tokenizer (no merges)")

        _model = model
        _tokenizer = tokenizer
        _state["ready"] = True
        _state["trained"] = trained
        _state["error"] = None
        print("[slm] Ready")
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
        "error": _state["error"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model = _model
    tokenizer = _tokenizer
    trained = _state["trained"]
    ready = _state["ready"]
    err = _state["error"]
    if not ready or model is None or tokenizer is None:
        detail = err or "ARCANE SLM is still loading"
        raise HTTPException(status_code=503, detail=detail)

    prompt = _build_prompt(req.history, req.message)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    allowed = tokenizer.generation_ids(printable_only=True)
    try:
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=20,
            eos_id=EOS_ID,
            allowed_token_ids=allowed,
        )
        reply = tokenizer.decode(out_ids[len(prompt_ids) :]).strip()
        if not reply:
            reply = "…"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return ChatResponse(reply=reply, trained=trained, preset=PRESET)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
