"""Tests for the Graph of Thought harness: document graph, retrieval, reasoning and tools."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.got import DocumentGraph, GroundedGraphOfThought, got_tools, parse_sections
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


def test_link_targets_are_not_indexed():
    g = DocumentGraph()
    g.add_document("# Notes\n\n## Links\n- [guide](distillation-guide.md)\n\n## Distillation\nTrain a student.", "Notes")
    assert all(h["lexicalScore"] == 0 for h in g.search("distillation") if h["title"] == "Links")


def test_support_is_checked_without_an_llm():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    share, nodes, _ = g.support("Edit config.json and set DATABASE_URL.", list(g.nodes))
    assert share == 1.0 and set(nodes) == {"user-guide#2", "user-guide#3"}  # one claim, two sections
    assert g.support("Reinstall the kubernetes helm chart.", list(g.nodes))[0] == 0.0


def scripted(*replies):
    """LLM stub that returns ``replies`` in order and records the prompts."""
    prompts, queue = [], list(replies)

    def llm(prompt):
        prompts.append(prompt)
        return queue.pop(0)
    return llm, prompts


GROUNDED, DB = "Edit config.json to customize settings.", "Edit config.json. Set DATABASE_URL before starting."
HALLUCINATED = "Reinstall the kubernetes helm chart."


def test_ungrounded_candidates_are_pruned_before_merging():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    llm, _ = scripted(json.dumps({"thoughts": [{"answer": GROUNDED}, {"answer": HALLUCINATED}]}))
    out = GroundedGraphOfThought(g, llm).reason("config fails, what now?")
    assert out["answer"] == GROUNDED and out["unsupported"] == []
    assert out["llm_calls"] == 1  # no judge calls, and nothing left to merge or repair
    assert [c["nodeId"] for c in out["citations"]] == ["user-guide#2"]


def test_merge_that_adds_unsupported_claims_is_rejected():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    llm, _ = scripted(json.dumps({"thoughts": [{"answer": GROUNDED}, {"answer": DB}]}),
                      json.dumps({"answer": f"{DB} {HALLUCINATED}"}))
    out = GroundedGraphOfThought(g, llm).reason("config fails, what now?")
    assert out["answer"] == DB and out["llm_calls"] == 2
    assert any(t["operation"] == "aggregate" for t in out["thoughts"])  # recorded, not chosen
    assert {c["nodeId"] for c in out["citations"]} == {"user-guide#2", "user-guide#3"}


def test_unsupported_claim_is_grounded_by_searching_for_it():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    llm, _ = scripted(json.dumps({"thoughts": [{"answer": DB}]}))
    out = GroundedGraphOfThought(g, llm, max_results=2).reason("config fails, what now?")
    assert "user-guide#3" not in out["thoughts"][0]["evidence"]  # the first search missed Database
    assert out["best"]["operation"] == "ground" and out["unsupported"] == []
    assert out["llm_calls"] == 1  # grounded by a graph search, not a rewrite


def test_follow_up_evidence_reaches_the_prompt_and_budget_is_reported():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")
    first = json.dumps({"thoughts": [{"answer": HALLUCINATED, "missing": "DATABASE_URL"}]})
    llm, prompts = scripted(first, json.dumps({"answer": "Set DATABASE_URL before starting."}))
    out = GroundedGraphOfThought(g, llm, max_results=2, max_evidence=1).reason("config fails, what now?")
    assert "[user-guide#3]" in prompts[1]  # newest evidence comes first, even with one evidence slot
    assert out["answer"] == "Set DATABASE_URL before starting." and not out["budget_exhausted"]

    llm, _ = scripted(first)
    capped = GroundedGraphOfThought(g, llm, max_results=2, max_llm_calls=1).reason("config fails, what now?")
    assert capped["budget_exhausted"] and capped["unsupported"] == [HALLUCINATED]


def test_empty_off_topic_and_contradicted_answers_lose():
    g = DocumentGraph()
    g.add_document(GUIDE, "User Guide")

    def reason(*answers, **kwargs):
        llm, _ = scripted(json.dumps({"thoughts": [{"answer": a} for a in answers]}))
        return GroundedGraphOfThought(g, llm, max_llm_calls=1, **kwargs).reason("config fails, what now?")

    partial = f"{GROUNDED} {HALLUCINATED}"
    assert reason("It is what it is.", partial)["answer"] == partial  # nothing to check is not "fully grounded"
    off_topic = "Push the build to the server. Run npm install to get started."  # more grounded mass, wrong sections
    assert reason(off_topic, GROUNDED)["answer"] == GROUNDED

    def nli(claim, evidence):  # stand-in for an entailment model
        return 0.0 if "never" in claim.lower() else 1.0
    assert reason("Never edit config.json.", verify=nli)["unsupported"] == ["Never edit config.json."]
    assert reason("Never edit config.json.")["unsupported"] == []  # the lexical check alone misses it


def test_tools_run_through_arcane_executor():
    g = DocumentGraph()
    assert "got_remove_document" not in [t.name for t in got_tools(g)]  # write tools are opt-in
    tools = got_tools(g, GroundedGraphOfThought(g, lambda p: '{"answer": "x"}'), writable=True)
    assert [t.name for t in tools][-1] == "got_reason"
    calls = [{"name": "got_add_document", "arguments": {"content": GUIDE, "title": "User Guide"}},
             {"name": "got_search", "arguments": {"query": "deploy"}},
             {"name": "got_read_node", "arguments": {"node_id": "nope"}}]
    added, hits, missing = execute_tools(calls, tools)
    assert added == "user-guide" and hits[0]["title"] == "Deployment"
    assert missing["ok"] is False and "Unknown node" in missing["error"]
    assert tools[0].schema_dict()["parameters"]["required"] == ["query"]
