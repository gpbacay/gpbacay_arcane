#!/usr/bin/env python3
"""Size and held-out accuracy of ARC 1 in every .rcn weight format.

Writes Models/arc1_rcn_benchmark.json (read by the docs site's size chart).

  python examples/benchmark_rcn.py
"""
import os, sys, json, tempfile, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from gpbacay_arcane.arc1 import Arc1Config, Arc1Model
from gpbacay_arcane.tokenization import BytePairTokenizer
from gpbacay_arcane.tools import Arc1Agent
from gpbacay_arcane.rcn import save_rcn, load_rcn
from gpbacay_arcane.arc1_data import fixed_eval_set, fixed_extract_set, fixed_classify_set
from gpbacay_arcane.arc1_train import evaluate_tools, evaluate_extract, evaluate_classify

OUT = tempfile.mkdtemp()
cfg = Arc1Config.from_dict(json.load(open("Models/arc1_arc1_tiny.config.json")))
base = Arc1Model(cfg).build_model(); base.load_weights("Models/arc1_arc1_tiny.weights.h5")
tok = BytePairTokenizer.load("Models/arc1_arc1_tiny_tokenizer.json")
tools, ext, cls = fixed_eval_set("eval", 150), fixed_extract_set(100), fixed_classify_set("eval", 150)

def score(model, t):
    a = Arc1Agent(model, t)
    r = evaluate_tools(a, tools); e = evaluate_extract(a, ext); c = evaluate_classify(a, cls)
    return {"exact_call": round(r["exact_call_acc"], 3), "extract_f1": round(e["field_f1"], 3), "classify": round(c["accuracy"], 3)}

res = {"weights.h5 (f32)": {"bytes": os.path.getsize("Models/arc1_arc1_tiny.weights.h5"), **score(base, tok)}}
for q in ("f16", "rq8", "rq4"):
    path = os.path.join(OUT, f"arc1-tiny-{q}.rcn")
    info = save_rcn(base, tok, path, quant=q)
    t0 = time.perf_counter(); m, t, head = load_rcn(path); load_s = time.perf_counter() - t0
    res[q] = {"bytes": info["bytes"], "load_s": round(load_s, 2), **score(m, t)}
    print(q, res[q], flush=True)
print(json.dumps(res, indent=1))
res["tflite int8"] = {"bytes": os.path.getsize("Models/arc1_export/arc1_int8.tflite")} if os.path.exists("Models/arc1_export/arc1_int8.tflite") else None
with open("Models/arc1_rcn_benchmark.json", "w") as f:
    json.dump(res, f, indent=1)
