"""Tests for the Graph of Thought harness: document graph, retrieval, reasoning and tools."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.got import DocumentGraph, GraphOfThought, got_tools, ops, parse_sections
from gpbacay_arcane.tools import execute_tools

GUIDE = """# User Guide

Intro text.

## Installation
Run npm install to get started.

## Configuration
Edit config.json to customize settings.

```
# not a heading
```

### Database
Set DATABASE_URL before starting.

## Deployment
Push the build to the server.

## Troubleshooting
If config fails, see the Configuration section.
"""


def titles(hits):
    return [h["title"] for h in hits]


def test_sections_and_structure():
    assert [s[0] for s in parse_sections(GUIDE)] == ["User Guide", "Installation", "Configuration", "Database", "Deployment", "Troubleshooting"]
    g = DocumentGraph()
    doc = g.add_document(GUIDE, "User Guide")
    assert g.nodes["user-guide"]["content"] == "Intro text."
    assert "  - [user-guide#3] Database" in g.outline(doc["docId"])
    assert "# not a heading" in g.nodes["user-guide#2"]["content"]


def test_retrieval_follows_references_and_stems():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    hits = g.search("config fails")
    assert hits[0]["title"] == "Troubleshooting"
    conf = next(h for h in hits if h["title"] == "Configuration")
    assert conf["edgeType"] == "reference" and conf["path"] == ["user-guide#5", "user-guide#2"]
    assert titles(g.search("deploy"))[0] == "Deployment"
    assert g.search("zebra giraffe") == []
    assert g.retrieve("deploy").startswith("### Deployment [user-guide#4]")


def test_replace_remove_and_json_roundtrip():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    g.add_document("# FAQ\n\n## Refunds\nMoney back within 30 days.", "FAQ")
    g.add_document("# FAQ\n\n## Refunds\nNo refunds.", "FAQ")  # replaces
    assert titles(g.search("refund")) == ["Refunds"] and "No refunds." in g.retrieve("refund")
    restored = DocumentGraph.from_json(json.loads(json.dumps(g.to_json())))
    assert titles(restored.search("config fails")) == titles(g.search("config fails"))
    assert g.remove_document("faq") and g.search("refund") == []
    assert not any("faq" in nid for links in g.adj.values() for nid in links)


def test_embedding_seeds():
    # Fake embedder: "deploy"-ish texts share a direction no lexical match would find
    def embed(text):
        return np.array([1.0, 0.0]) if "ship" in text.lower() or "push" in text.lower() else np.array([0.0, 1.0])

    g = DocumentGraph(embed=embed)
    g.add_document(GUIDE, "User Guide")
    assert titles(g.search("how do I ship it"))[0] == "Deployment"


def test_reasoner_merges_and_follows_up():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        if "Propose 2" in prompt:
            return '{"thoughts": [{"answer": "Edit config.json", "evidence": ["user-guide#2"]}, {"answer": "Reinstall"}]}'
        if "Rate the candidate" in prompt:
            return '{"score": 8, "critique": "ok"}' if "config.json" in prompt.split("Candidate answer:")[1] else '{"score": 2}'
        return '{"answer": "Edit config.json and set DATABASE_URL", "evidence": ["user-guide#3"], "missing": ""}'

    plan = [ops.retrieve(), ops.generate(2), ops.score(), ops.keep_best(2), ops.aggregate(), ops.refine(1), ops.score()]
    out = GraphOfThought(g, llm).reason("config fails, what now?", plan)
    assert out["answer"] == "Edit config.json and set DATABASE_URL"
    assert {"user-guide#2", "user-guide#3"} <= {c["nodeId"] for c in out["citations"]}
    assert out["llm_calls"] == len(prompts) == 6  # generate 1, score 2, aggregate 1, refine 1, score 1
    assert out["best"]["operation"] == "refine" and out["best"]["score"] == 0.8
    capped = GraphOfThought(g, llm, max_llm_calls=1).reason("config fails")
    assert capped["llm_calls"] == 1


def test_tools_run_through_arcane_executor():
    g = DocumentGraph()
    assert "got_remove_document" not in [t.name for t in got_tools(g)]  # write tools are opt-in
    tools = got_tools(g, GraphOfThought(g, lambda p: '{"answer": "x"}'), writable=True)
    assert [t.name for t in tools][-1] == "got_reason"
    calls = [{"name": "got_add_document", "arguments": {"content": GUIDE, "title": "User Guide"}},
             {"name": "got_search", "arguments": {"query": "deploy"}},
             {"name": "got_read_node", "arguments": {"node_id": "nope"}}]
    added, hits, missing = execute_tools(calls, tools)
    assert added == "user-guide" and hits[0]["title"] == "Deployment"
    assert missing["ok"] is False and "Unknown node" in missing["error"]
    assert tools[0].schema_dict()["parameters"]["required"] == ["query"]
