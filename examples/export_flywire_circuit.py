"""
Refresh synapse counts on the bundled FlyWire circuit using fafbseg-py.

Requires a FlyWire CAVE token:
  https://fafbseg-py.readthedocs.io/en/stable/source/tutorials/flywire_setup.html

Usage (from repo root):
  python examples/export_flywire_circuit.py --live
"""
from __future__ import annotations

import argparse
import json
import os
import sys

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
DEFAULT_PATH = os.path.join(ROOT_DIR, "data", "flywire_escape_circuit.json")


def refresh_with_fafbseg(circuit: dict) -> dict:
    try:
        from fafbseg import flywire
    except ImportError as exc:
        raise SystemExit(
            "fafbseg is not installed. pip install fafbseg\n"
            "Then store a CAVE token as described in the fafbseg FlyWire setup tutorial."
        ) from exc

    ids = [int(n["id"]) for n in circuit["neurons"]]
    flywire.set_default_dataset("public")
    conn = flywire.get_connectivity(ids)
    edges = []
    idset = {str(i) for i in ids}
    for row in conn.itertuples(index=False):
        pre = str(int(getattr(row, "pre")))
        post = str(int(getattr(row, "post")))
        weight = float(getattr(row, "weight", getattr(row, "syn_count", 0)))
        if pre in idset and post in idset and pre != post and weight >= 5:
            edges.append({"pre": pre, "post": post, "synapses": weight, "sign": 1})
    circuit["edges"] = edges
    circuit["tools"] = ["fafbseg-py live query", "FlyWire public materialization 783"]
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Query FlyWire via fafbseg")
    parser.add_argument("--path", default=DEFAULT_PATH)
    args = parser.parse_args()
    with open(args.path, encoding="utf-8") as handle:
        circuit = json.load(handle)
    if args.live:
        circuit = refresh_with_fafbseg(circuit)
        with open(args.path, "w", encoding="utf-8") as handle:
            json.dump(circuit, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(circuit['neurons'])} neurons, {len(circuit['edges'])} edges to {args.path}")
    else:
        print(
            f"Snapshot already at {args.path} "
            f"({len(circuit['neurons'])} neurons, {len(circuit['edges'])} edges)."
        )
        print("Pass --live to refresh synapses with fafbseg.")


if __name__ == "__main__":
    sys.exit(main())
