"""ARC 1 over stdin/stdout, one JSON object per line. Used by the Node.js package.

    python -m gpbacay_arcane.arc1_stdio [--model path.rcn]

Request:  {"id": 1, "method": "run" | "extract" | "classify" | "embed", "params": {...}}
Response: {"id": 1, "result": ...} or {"id": 1, "error": "..."}
The first line written is {"ready": true} once the model is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from .tools import ToolParam, ToolSpec


def tools_from_dicts(items: List[Dict[str, Any]]) -> List[ToolSpec]:
    """``[{"name", "description", "parameters": [{"name", "type", "description", "required", "enum"}]}]``"""
    return [
        ToolSpec(
            name=t["name"],
            description=t.get("description", ""),
            parameters=[
                ToolParam(
                    name=p["name"],
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", True),
                    enum=p.get("enum"),
                    enum_descriptions=p.get("enum_descriptions"),
                )
                for p in t.get("parameters", [])
            ],
        )
        for t in items
    ]


def handle(agent, method: str, params: Dict[str, Any]) -> Any:
    if method == "run":
        return agent.run(params["prompt"], tools=tools_from_dicts(params.get("tools") or []),
                         execute=False, cycles=params.get("cycles"))
    if method == "extract":
        return agent.extract(params["text"], params["schema"], cycles=params.get("cycles"))
    if method == "classify":
        return agent.classify(params["text"], params["labels"], task=params.get("task"),
                              cycles=params.get("cycles"), descriptions=params.get("descriptions"))
    if method == "embed":
        return agent.embed(params["text"])
    raise ValueError(f"unknown method {method!r}")


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to an .rcn file (default: bundled arc1-tiny)")
    args = ap.parse_args()

    # Protocol owns stdout; anything else (TensorFlow logs, prints) goes to stderr.
    out, sys.stdout = sys.stdout, sys.stderr

    def send(obj: Dict[str, Any]) -> None:
        out.write(json.dumps(obj, default=_json_default) + "\n")
        out.flush()

    from .rcn import load_arc1

    agent = load_arc1(args.model)
    send({"ready": True})
    for line in sys.stdin:
        if not line.strip():
            continue
        req: Dict[str, Any] = {}
        try:
            req = json.loads(line)
            send({"id": req.get("id"), "result": handle(agent, req["method"], req.get("params") or {})})
        except Exception as exc:  # noqa: BLE001 — report to the caller, keep serving
            send({"id": req.get("id"), "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
