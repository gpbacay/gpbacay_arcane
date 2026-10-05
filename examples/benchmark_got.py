#!/usr/bin/env python3
"""HotpotQA (distractor) benchmark: plain RAG vs LLM-judge GoT vs Grounded GoT vs Grounded GoT + NLI.

All four arms share one DocumentGraph per question (the 10 paragraphs), one retrieval setting and one
local LLM at temperature 0. The LLM-judge arm is the pre-grounding GoT (commit 75f01ee). Hallucination is
scored by an NLI model *different* from the one used by the ``verify=`` arm, against the gold paragraphs.

  python examples/benchmark_got.py run  --data hotpot_dev.json --llm qwen2.5-1.5b-instruct-q4_k_m.gguf -n 100
  python examples/benchmark_got.py eval --data hotpot_dev.json
"""
import argparse, collections, json, os, random, re, string, subprocess, sys, time, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

OUT = os.path.join(ROOT, "Models", "got_benchmark")
ARMS = ["rag", "judge_got", "grounded_got", "grounded_got_nli"]
VERIFY_NLI, EVAL_NLI = "cross-encoder/nli-deberta-v3-small", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
JUDGE_COMMIT = "75f01ee"
TOP_K, EVID_CHARS = 6, 700
STYLE = "Answer in one or two complete sentences."
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
ABSTAIN = re.compile(r"(not (mention|provide|contain|specif|state|enough|in the evidence)|no (information|evidence)|cannot (be )?(determin|answer|find))", re.I)


# ----------------------------------------------------------------------------- scoring helpers
def norm_tokens(s):
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    return [w for w in s.split() if w not in ("a", "an", "the")]


def contains(gold, pred):
    g, p = norm_tokens(gold), norm_tokens(pred)
    return bool(g) and any(p[i:i + len(g)] == g for i in range(len(p) - len(g) + 1))


def f1(gold, pred):
    g, p = norm_tokens(gold), norm_tokens(pred)
    common = sum((collections.Counter(g) & collections.Counter(p)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def clean_answer(text):
    """A JSON blob that the model failed to unwrap counts as its ``answer`` field if parseable."""
    t = (text or "").strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            return str(d.get("answer", "")), False
        except json.JSONDecodeError:
            return t, True
    return t, False


class NLI:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.ent = next(i for i, l in self.model.config.id2label.items() if l.lower().startswith("entail"))

    def __call__(self, claim, premise):
        enc = self.tok(premise, claim, truncation="only_first", max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            return float(self.model(**enc).logits.softmax(-1)[0, self.ent])


# ----------------------------------------------------------------------------- data
def load_questions(path, n, seed):
    data = json.load(open(path, encoding="utf-8"))
    random.Random(seed).shuffle(data)
    return data[:n]


def build(cls, ex):
    g = cls()
    for i, (title, sents) in enumerate(ex["context"]):
        g.add_document(" ".join(s.strip() for s in sents), title, doc_id=f"d{i}")
    return g


def gold_titles(ex):
    return {t for t, _ in ex["supporting_facts"]}


def gold_premise(ex):
    return "\n".join(" ".join(s.strip() for s in sents) for t, sents in ex["context"] if t in gold_titles(ex))


# ----------------------------------------------------------------------------- LLM (cached, timed, counted)
class LLM:
    def __init__(self, path):
        from llama_cpp import Llama
        self.m = Llama(model_path=path, n_ctx=6144, n_threads=os.cpu_count(), verbose=False)
        self.cache, self.spent = {}, 0.0

    def __call__(self, prompt):
        if prompt not in self.cache:
            t0 = time.perf_counter()
            kw = {"response_format": {"type": "json_object"}} if "JSON only" in prompt else {}
            r = self.m.create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.0,
                                              max_tokens=450, **kw)
            self.cache[prompt] = (r["choices"][0]["message"]["content"], time.perf_counter() - t0)
        reply, secs = self.cache[prompt]
        self.spent += secs  # logical latency: a cache hit still costs what the call originally cost
        return reply


def judge_module():
    """The LLM-judge GoT as it was before grounding, loaded from git."""
    src = subprocess.check_output(["git", "show", f"{JUDGE_COMMIT}:gpbacay_arcane/got.py"], cwd=ROOT).decode()
    src = src.replace("from .tools import", "from gpbacay_arcane.tools import")
    mod = types.ModuleType("got_judge")
    sys.modules["got_judge"] = mod
    exec(compile(src, "got_judge.py", "exec"), mod.__dict__)
    return mod


# ----------------------------------------------------------------------------- run
def run(args):
    import gpbacay_arcane.got as new
    old = judge_module()
    llm, verify = LLM(args.llm), NLI(VERIFY_NLI)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "answers.jsonl")
    done = {(r["id"], r["arm"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    kw = dict(max_evidence=TOP_K, max_evidence_chars=EVID_CHARS, max_results=TOP_K)

    def arm_rag(ex, g):
        ids = [h["nodeId"] for h in g.search(ex["question"], max_results=TOP_K)]
        ctx = "\n\n".join(f"[{i}] {g.nodes[i]['title']}\n{g.nodes[i]['content'][:EVID_CHARS]}" for i in ids)
        out = llm(f"Question: {ex['question']}\n\nEvidence:\n{ctx}\n\nAnswer only from the evidence. {STYLE}")
        return {"answer": out.strip(), "llm_calls": 1, "retrieved": ids}

    def arm_got(cls_mod, **extra):
        def go(ex, g):
            gg = build(cls_mod.DocumentGraph, ex) if cls_mod is old else g
            cls = old.GraphOfThought if cls_mod is old else new.GroundedGraphOfThought
            res = cls(gg, llm, **kw, **extra).reason(ex["question"] + " " + STYLE)
            return {"answer": res["answer"], "llm_calls": res["llm_calls"],
                    "retrieved": [h["nodeId"] for h in gg.search(ex["question"], max_results=TOP_K)],
                    "unsupported_self": res.get("unsupported")}
        return go

    runners = {"rag": arm_rag, "judge_got": arm_got(old), "grounded_got": arm_got(new),
               "grounded_got_nli": arm_got(new, verify=verify)}
    with open(path, "a") as f:
        for qi, ex in enumerate(load_questions(args.data, args.n, args.seed)):
            g = build(new.DocumentGraph, ex)
            for arm in ARMS:
                if (ex["_id"], arm) in done:
                    continue
                t0, s0 = time.perf_counter(), llm.spent
                rec = runners[arm](ex, g)
                rec.update(id=ex["_id"], arm=arm, llm_seconds=round(llm.spent - s0, 2), wall=round(time.perf_counter() - t0, 2))
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"q{qi} {arm:17s} calls={rec['llm_calls']} llm_s={rec['llm_seconds']:.0f} wall={rec['wall']:.0f}s", flush=True)


# ----------------------------------------------------------------------------- eval
def boot(xs, reps=2000, seed=0):
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(xs, k=len(xs))) / len(xs) for _ in range(reps))
    return sum(xs) / len(xs), means[int(.025 * reps)], means[int(.975 * reps)]


def evaluate(args):
    qs = {ex["_id"]: ex for ex in load_questions(args.data, 10 ** 9, args.seed)}
    rows = [json.loads(l) for l in open(os.path.join(OUT, "answers.jsonl"))]
    complete = {i for i in qs if all(any(r["id"] == i and r["arm"] == a for r in rows) for a in ARMS)}
    nli, cache, per = NLI(EVAL_NLI), {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["id"] not in complete:
            continue
        ex = qs[r["id"]]
        ans, json_fail = clean_answer(r["answer"])
        sents = [s.strip() for s in SENT.split(ans) if len(s.split()) >= 3]
        abst = [s for s in sents if ABSTAIN.search(s)]
        checked = [s for s in sents if s not in abst]
        prem = gold_premise(ex)
        bad = [s for s in checked if cache.setdefault((r["id"], s), nli(s, prem)) < 0.5]
        m = per[r["arm"]]
        m["correct"].append(float(contains(ex["answer"], ans)))
        m["f1"].append(f1(ex["answer"], ans))
        m["halluc_sent_rate"].append(len(bad) / len(checked) if checked else 0.0)
        m["halluc_answer"].append(float(bool(bad)))
        m["abstain"].append(float(bool(abst) and not checked))
        m["json_fail"].append(float(json_fail))
        m["words"].append(len(ans.split()))
        m["llm_calls"].append(r["llm_calls"])
        m["llm_seconds"].append(r["llm_seconds"])
        m["gold_recall@k"].append(len(gold_titles(ex) & retrieved_titles(ex, r)) / len(gold_titles(ex)))
    n = len(complete)
    print(f"\nHotpotQA distractor dev, n={n} questions (seed {args.seed}); 95% bootstrap CI\n")
    cols = ["correct", "f1", "halluc_answer", "halluc_sent_rate", "abstain", "llm_calls", "llm_seconds", "words", "gold_recall@k", "json_fail"]
    print(f"{'arm':18s}" + "".join(f"{c:>16s}" for c in cols))
    summary = {}
    for arm in ARMS:
        summary[arm] = {c: boot(per[arm][c]) for c in cols if per[arm][c]}
        print(f"{arm:18s}" + "".join(f"{summary[arm][c][0]:>16.3f}" for c in cols))
        print(f"{'  95% CI':18s}" + "".join(f"{'[%.2f,%.2f]' % summary[arm][c][1:]:>16s}" for c in cols))
    json.dump({"n": n, "seed": args.seed, "summary": summary}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


def retrieved_titles(ex, r):
    """Titles of the paragraphs the arm's search surfaced (node ids are ``d<i>``)."""
    return {ex["context"][int(i[1:].split("#")[0])][0] for i in r["retrieved"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--llm")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else evaluate(a)
