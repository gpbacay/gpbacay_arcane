#!/usr/bin/env python3
"""Rebuild every served copy of ARC 1 from the trained weights, so all of them run the latest model.

  Models/arc1_arc1_tiny.{config.json,weights.h5}, _tokenizer.json   trained model (the docs demo server loads these)
    -> gpbacay_arcane/models/arc1-tiny.rcn   4-bit, bundled in the package (load_arc1(), the Node runtime)
    -> node/web/arc1.onnx + arc1.json        the browser runtime, and the docs site's chatbot

arc1.json records a fingerprint of the trained files. The docs site's build refuses a stale copy
(arcane-docs-web/scripts/copy-arc1-web.mjs), and .github/workflows/publish-arc1.yml runs this script on
every push that changes the trained files, so committing new weights is enough.

    pip install tensorflow tf2onnx onnxruntime
    python examples/publish_arc1.py           # rebuild the copies
    python examples/publish_arc1.py --check   # exit 1 if they are older than the weights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [os.path.join("Models", name) for name in
           ("arc1_arc1_tiny.config.json", "arc1_arc1_tiny.weights.h5", "arc1_arc1_tiny_tokenizer.json")]
WEB_JSON = os.path.join(ROOT, "node", "web", "arc1.json")


def fingerprint() -> str:
    """sha256 over the trained files. JSON line endings are normalized, since git may check them out as CRLF."""
    digest = hashlib.sha256()
    for rel in SOURCES:
        with open(os.path.join(ROOT, rel), "rb") as f:
            data = f.read()
        if rel.endswith(".json"):
            data = data.replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def recorded() -> str | None:
    with open(WEB_JSON, encoding="utf-8") as f:
        return json.load(f).get("source", {}).get("sha256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report whether the copies are stale")
    args = ap.parse_args()
    want = fingerprint()
    if args.check:
        stale = recorded() != want
        print(f"ARC 1 copies are {'STALE' if stale else 'up to date'} (weights {want[:12]})")
        sys.exit(1 if stale else 0)

    sys.path.insert(0, ROOT)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
    from gpbacay_arcane.rcn import ARC1_TINY, save_rcn
    from gpbacay_arcane.tokenization import BytePairTokenizer

    config_path, weights_path, tokenizer_path = (os.path.join(ROOT, rel) for rel in SOURCES)
    with open(config_path, encoding="utf-8") as f:
        model = Arc1Model(Arc1Config.from_dict(json.load(f)))
    model.build_model()
    model.load_weights(weights_path)
    info = save_rcn(model, BytePairTokenizer.load(tokenizer_path), ARC1_TINY, quant="rq4")
    print(f"RCN -> {ARC1_TINY} ({info['bytes']:,} bytes)")

    subprocess.run([sys.executable, os.path.join(ROOT, "examples", "export_arc1_onnx.py"),
                    "--model", ARC1_TINY, "--out", os.path.join(ROOT, "node", "web")], check=True)
    with open(WEB_JSON, encoding="utf-8") as f:
        web = json.load(f)
    web["source"] = {"sha256": want, "files": [rel.replace(os.sep, "/") for rel in SOURCES]}
    with open(WEB_JSON, "w", encoding="utf-8") as f:
        json.dump(web, f)
    print(f"stamped {WEB_JSON} with weights {want[:12]}")


if __name__ == "__main__":
    main()
