"""
FastAPI server: FlyWire neuron graph + ARCANE RSAA dynamics.

Run from repo root:
  python examples/serve_flywire_api.py

Render sets PORT. Optional live refresh:
  FLYWIRE_SECRET=<cave token>  (also accepts CAVE_TOKEN)
"""
from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CIRCUIT_PATH = os.environ.get(
    "FLYWIRE_CIRCUIT_PATH",
    os.path.join(ROOT_DIR, "data", "flywire_escape_circuit.json"),
)

import json

app = FastAPI(title="ARCANE FlyWire Connectome", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

circuit: dict | None = None
weight_matrix: np.ndarray | None = None
id_to_index: dict[str, int] = {}


class ResonateRequest(BaseModel):
    preset: Literal["loom", "walk", "backward", "custom"] = "loom"
    kind: Literal["resonant", "feedforward"] = "resonant"
    cycles: int = 8
    neuron_ids: list[str] = []


def load_circuit_file() -> dict:
    if not os.path.exists(CIRCUIT_PATH):
        raise RuntimeError(f"Circuit snapshot not found: {CIRCUIT_PATH}")
    with open(CIRCUIT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def build_weights(data: dict) -> tuple[np.ndarray, dict[str, int]]:
    index = {n["id"]: i for i, n in enumerate(data["neurons"])}
    n = len(data["neurons"])
    weights = np.zeros((n, n), dtype=np.float64)
    for edge in data["edges"]:
        if edge["pre"] not in index or edge["post"] not in index:
            continue
        i = index[edge["pre"]]
        j = index[edge["post"]]
        weights[j, i] += abs(float(edge["synapses"]))
    col_max = np.maximum(np.max(np.abs(weights), axis=0, keepdims=True), 1.0)
    weights = weights / col_max
    return weights, index


def try_live_refresh(data: dict) -> dict:
    token = os.environ.get("FLYWIRE_SECRET") or os.environ.get("CAVE_TOKEN")
    if not token:
        data["live"] = False
        return data
    try:
        sys.path.insert(0, EXAMPLES_DIR)
        from export_flywire_circuit import refresh_with_fafbseg

        refreshed = refresh_with_fafbseg(data)
        refreshed["live"] = True
        return refreshed
    except Exception as exc:
        data["live"] = False
        data["live_error"] = str(exc)
        return data


@app.on_event("startup")
def startup() -> None:
    global circuit, weight_matrix, id_to_index
    data = load_circuit_file()
    data = try_live_refresh(data)
    weight_matrix, id_to_index = build_weights(data)
    circuit = data
    print(f"Loaded {len(data['neurons'])} neurons, {len(data['edges'])} edges")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "neurons": len(circuit["neurons"]) if circuit else 0,
        "edges": len(circuit["edges"]) if circuit else 0,
        "live": bool(circuit and circuit.get("live")),
    }


@app.get("/circuit")
def get_circuit():
    if circuit is None:
        raise HTTPException(status_code=503, detail="Circuit not loaded")
    return circuit


def stimulus_vector(req: ResonateRequest) -> np.ndarray:
    assert circuit is not None
    n = len(circuit["neurons"])
    stim = np.zeros(n, dtype=np.float64)
    if req.preset == "custom" and req.neuron_ids:
        for nid in req.neuron_ids:
            if nid in id_to_index:
                stim[id_to_index[nid]] = 1.0
        return stim
    for i, neuron in enumerate(circuit["neurons"]):
        ct = neuron["cell_type"]
        if req.preset == "loom" and ct in ("LC4", "LPLC2"):
            stim[i] = 1.0
        elif req.preset == "walk" and ct in ("DNp09", "DNa01", "DNa02"):
            stim[i] = 1.0
        elif req.preset == "backward" and ct == "MDN":
            stim[i] = 1.0
    return stim


def rsaa_step(stim: np.ndarray, kind: str, cycles: int) -> dict:
    assert circuit is not None and weight_matrix is not None
    neurons = circuit["neurons"]
    sensory = np.array([n["layer"] == "sensory" for n in neurons])
    descending = np.array([n["layer"] == "descending" for n in neurons])
    state = stim.copy()
    n_cycles = 1 if kind == "feedforward" else max(1, min(24, int(cycles)))
    gamma = 0.42 if kind == "resonant" else 0.08
    leak = 0.35
    last_div = 0.0
    for _ in range(n_cycles):
        incoming = np.tanh(weight_matrix @ state)
        if kind == "resonant" and sensory.any() and descending.any():
            intent = float(np.mean(state[descending]))
            projected = np.tanh(weight_matrix.T @ state)
            target = 0.65 * projected + 0.35 * intent
            state[sensory] += gamma * (target[sensory] - state[sensory])
            last_div = float(np.sqrt(np.mean((state[sensory] - target[sensory]) ** 2)))
        state = (1.0 - leak) * state + leak * incoming
        state = np.tanh(state)
        state = np.maximum(state, stim * 0.8)

    activity = {neurons[i]["id"]: round(float(state[i]), 4) for i in range(len(neurons))}
    by_type: dict[str, list[float]] = {}
    for i, neuron in enumerate(neurons):
        by_type.setdefault(neuron["cell_type"], []).append(float(state[i]))
    type_mean = {k: round(float(np.mean(v)), 4) for k, v in by_type.items()}
    gf = [float(state[i]) for i, n in enumerate(neurons) if n["cell_type"] == "DNp01"]
    return {
        "activity": activity,
        "type_mean": type_mean,
        "giant_fiber": round(float(np.mean(gf) if gf else 0.0), 4),
        "divergence": round(last_div, 4),
        "cycles": n_cycles,
        "kind": kind,
    }


@app.post("/resonate")
def resonate(req: ResonateRequest):
    if circuit is None or weight_matrix is None:
        raise HTTPException(status_code=503, detail="Circuit not loaded")
    stim = stimulus_vector(req)
    if not np.any(stim):
        raise HTTPException(status_code=400, detail="Stimulus was empty")
    return rsaa_step(stim, req.kind, req.cycles)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
