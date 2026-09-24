#!/usr/bin/env python3
"""FastAPI server for ARC 1 tool calling, extraction, and embeddings.

Used by the docs site at /docs/arc-1.

Env:
  ARC1_PRESET=arc1-tiny|arc1
  ARC1_CONFIG_PATH=Models/arc1_arc1_tiny.config.json
  ARC1_WEIGHTS_PATH=Models/arc1_arc1_tiny.weights.h5
  ARC1_TOKENIZER_PATH=Models/arc1_arc1_tiny_tokenizer.json
  ARC1_METRICS_PATH=Models/arc1_arc1_tiny.metrics.json
  ARC1_ALLOW_HEURISTIC=0   # 1 = keyword fallback when the model selects nothing
  PORT=8002
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.tokenization import BASE_VOCAB, BytePairTokenizer
from gpbacay_arcane.tools import Arc1Agent, ToolParam, ToolSpec

app = FastAPI(title="ARCANE ARC 1", version="0.1.0")
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


DEFAULT_CONFIG = os.path.join(MODELS_DIR, "arc1_arc1_tiny.config.json")
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "arc1_arc1_tiny.weights.h5")
DEFAULT_TOK = os.path.join(MODELS_DIR, "arc1_arc1_tiny_tokenizer.json")

CONFIG_PATH = _env("ARC1_CONFIG_PATH") or _first_existing(DEFAULT_CONFIG)
PRESET = (_env("ARC1_PRESET") or ("arc1-tiny" if not CONFIG_PATH else "arc1-tiny")).lower()
WEIGHTS_PATH = _env("ARC1_WEIGHTS_PATH") or _first_existing(
    DEFAULT_WEIGHTS if CONFIG_PATH else "",
    os.path.join(MODELS_DIR, "arc1_arc1_tiny.weights.h5"),
)
TOKENIZER_PATH = _env("ARC1_TOKENIZER_PATH") or _first_existing(
    DEFAULT_TOK,
    os.path.join(MODELS_DIR, "arc1_arc1_tiny_tokenizer.json"),
)

METRICS_PATH = _env("ARC1_METRICS_PATH") or _first_existing(
    os.path.join(MODELS_DIR, "arc1_arc1_tiny.metrics.json"),
)
ALLOW_HEURISTIC = _env("ARC1_ALLOW_HEURISTIC") in ("1", "true", "yes")

_model: Optional[Arc1Model] = None
_tokenizer: Any = None
_agent: Optional[Arc1Agent] = None
_state = {
    "ready": False,
    "trained": False,
    "error": None,
    "preset": PRESET,
    "parameters": None,
    "active_depth": None,
    "ladder_depths": None,
    "calibration": None,
    "metrics": None,
}


def _builtin_tools() -> List[ToolSpec]:
    def get_weather(city: str):
        """Get the current weather for a city."""
        temps = {"Lagos": 27, "Manila": 31, "Tokyo": 18, "Paris": 14, "Berlin": 12, "London": 11}
        return {"city": city, "temp_c": temps.get(city, 22), "sky": "clear"}

    def set_lights(room: str, level: int = 50):
        """Set light brightness in a room."""
        return {"room": room, "level": int(level), "ok": True}

    def convert_currency(amount: float, from_currency: str, to_currency: str):
        """Convert an amount between currencies using demo rates."""
        # Demo rates vs USD — not live FX.
        usd = {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 151.0,
            "PHP": 56.5,
            "NGN": 1600.0,
        }
        src = str(from_currency).upper()
        dst = str(to_currency).upper()
        if src not in usd or dst not in usd:
            return {"ok": False, "error": f"Unsupported currency pair {src}->{dst}"}
        converted = float(amount) * (usd[dst] / usd[src])
        return {
            "amount": float(amount),
            "from": src,
            "to": dst,
            "converted": round(converted, 4),
            "rate": round(usd[dst] / usd[src], 6),
        }

    def send_message(to: str, message: str):
        """Send a short message to a contact."""
        return {"to": to, "message": message, "status": "queued"}

    return [
        ToolSpec(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters=[ToolParam(name="city", description="City name")],
            handler=get_weather,
        ),
        ToolSpec(
            name="set_lights",
            description="Set light brightness in a room (0-100).",
            parameters=[
                ToolParam(name="room", description="Room name"),
                ToolParam(name="level", type="integer", description="Brightness 0-100"),
            ],
            handler=set_lights,
        ),
        ToolSpec(
            name="convert_currency",
            description="Convert an amount from one currency to another.",
            parameters=[
                ToolParam(name="amount", type="number", description="Amount to convert"),
                ToolParam(name="from_currency", description="Source currency code, e.g. USD"),
                ToolParam(name="to_currency", description="Target currency code, e.g. PHP"),
            ],
            handler=convert_currency,
        ),
        ToolSpec(
            name="send_message",
            description="Send a short message to a contact.",
            parameters=[
                ToolParam(name="to", description="Contact name or handle"),
                ToolParam(name="message", description="Message body"),
            ],
            handler=send_message,
        ),
    ]


def _tools_from_payload(payload: Optional[List[Dict[str, Any]]]) -> List[ToolSpec]:
    if not payload:
        return _builtin_tools()
    tools = []
    builtins = {t.name: t for t in _builtin_tools()}
    for item in payload:
        name = item.get("name")
        if not name:
            continue
        params = [
            ToolParam(
                name=p.get("name", "arg"),
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                enum=p.get("enum"),
            )
            for p in item.get("parameters", [])
        ]
        handler = builtins[name].handler if name in builtins else None
        tools.append(
            ToolSpec(
                name=name,
                description=item.get("description", ""),
                parameters=params,
                handler=handler,
            )
        )
    return tools or _builtin_tools()


def _load_config() -> Arc1Config:
    if CONFIG_PATH:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[arc1] Loaded config {CONFIG_PATH}")
        return Arc1Config.from_dict(payload)
    print(f"[arc1] Using preset '{PRESET}'")
    return Arc1Config.from_preset(PRESET)


def _load_model() -> None:
    global _model, _tokenizer, _agent
    try:
        config = _load_config()
        model = Arc1Model(config)
        model.build_model()
        trained = False
        weight_note = None
        if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
            try:
                model.load_weights(WEIGHTS_PATH)
                trained = True
                print(f"[arc1] Loaded weights {WEIGHTS_PATH}")
            except Exception as exc:  # noqa: BLE001
                # Older checkpoints (e.g. confidence_head-only) do not match noul/choice/span.
                try:
                    model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
                    weight_note = f"partial load (skip_mismatch): {exc}"
                    print(f"[arc1] {weight_note}")
                except Exception as exc2:  # noqa: BLE001
                    weight_note = f"weight load failed, serving untrained: {exc2}"
                    print(f"[arc1] {weight_note}")
        else:
            print("[arc1] No weights found; serving an UNTRAINED model (train with examples/train_arc1.py)")

        if TOKENIZER_PATH and os.path.exists(TOKENIZER_PATH):
            tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
            print(f"[arc1] Loaded tokenizer {TOKENIZER_PATH}")
        else:
            tokenizer = BytePairTokenizer(vocab_size=max(config.vocab_size, BASE_VOCAB))
            print("[arc1] Using byte-level tokenizer")

        # Keyword fallback keeps the docs sandbox usable when decision heads are untrained.
        use_heuristic = ALLOW_HEURISTIC or not trained
        if use_heuristic and not ALLOW_HEURISTIC:
            print("[arc1] Enabling heuristic fallback (model not fully trained)")
        agent = Arc1Agent(
            model, tokenizer, tools=_builtin_tools(), allow_heuristic=use_heuristic
        )
        metrics = None
        if METRICS_PATH and os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
        _model = model
        _tokenizer = tokenizer
        _agent = agent
        _state.update(
            ready=True,
            trained=trained,
            error=weight_note,
            parameters=int(model.count_params()),
            active_depth=config.resolve_depth(),
            ladder_depths=sorted(set(config.ladder_depths) | {config.num_layers}),
            calibration=dict(config.calibration),
            metrics=_metrics_summary(metrics),
            preset=PRESET if not CONFIG_PATH else "from-config",
            heuristic_fallback=use_heuristic,
        )
        print(f"[arc1] Ready ({_state['parameters']:,} params, trained={trained})")
    except Exception as exc:  # noqa: BLE001
        _state["ready"] = False
        _state["error"] = str(exc)
        print(f"[arc1] Failed: {exc}")


def _metrics_summary(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Headline held-out numbers at full depth for /health."""
    if not metrics:
        return None
    evaluation = metrics.get("evaluation", {})
    depth_keys = sorted((k for k in evaluation if k.startswith("depth_")), key=lambda k: int(k.split("_")[1]))
    if not depth_keys:
        return None
    full = evaluation[depth_keys[-1]]
    return {
        "depth": int(depth_keys[-1].split("_")[1]),
        "exact_call_acc_heldout_values": full["tools_heldout_values"]["exact_call_acc"],
        "exact_call_acc_unseen_tools": full["tools_unseen_tools"]["exact_call_acc"],
        "extraction_field_f1": full["extraction_heldout_values"]["field_f1"],
        "noul_ece_after_calibration": metrics.get("calibration", {}).get("noul_ece_after"),
    }


def _check_depth(depth: Optional[int]) -> Optional[int]:
    if depth is None or _model is None:
        return None
    layers = _model.arc1_config.num_layers
    if not 1 <= int(depth) <= layers:
        raise HTTPException(status_code=422, detail=f"depth must be in 1..{layers}")
    return int(depth)


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    tools: Optional[List[Dict[str, Any]]] = None
    execute: bool = True
    depth: Optional[int] = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    schema: Dict[str, Any] = Field(default_factory=dict)
    depth: Optional[int] = None


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    depth: Optional[int] = None


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
        "parameters": _state["parameters"],
        "active_depth": _state["active_depth"],
        "ladder_depths": _state["ladder_depths"],
        "calibration": _state["calibration"],
        "metrics": _state["metrics"],
        "heuristic_fallback": _state.get("heuristic_fallback", ALLOW_HEURISTIC),
        "error": _state["error"],
        "model": "ARC 1",
    }


@app.post("/run")
def run(req: RunRequest):
    if not _state["ready"] or _agent is None or _model is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "ARC 1 still loading")
    depth = _check_depth(req.depth)
    tools = _tools_from_payload(req.tools)
    try:
        out = _agent.run(req.prompt, tools=tools, execute=req.execute, depth=depth)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"run failed: {exc}") from exc
    return out


@app.post("/extract")
def extract(req: ExtractRequest):
    if not _state["ready"] or _agent is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "ARC 1 still loading")
    schema = req.schema or {
        "name": {"type": "string", "description": "Person name"},
        "city": {"type": "string", "description": "City"},
    }
    depth = _check_depth(req.depth)
    try:
        return _agent.extract(req.text, schema, depth=depth)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"extract failed: {exc}") from exc


@app.post("/embed")
def embed(req: EmbedRequest):
    if not _state["ready"] or _agent is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "ARC 1 still loading")
    depth = _check_depth(req.depth)
    try:
        vec = _agent.embed(req.text, depth=depth)
        return {"embedding": vec, "dim": len(vec)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"embed failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
