"""Graph of Thought: a document knowledge graph, graph retrieval and Graph-of-Thoughts reasoning.

Python port of https://github.com/gpbacay/graph-of-thought (v2), wired into ARC 1:

- ``DocumentGraph(embed=agent.embed)`` links sections and seeds queries with ARC 1 embeddings
  (without ``embed`` it links by shared distinctive terms and needs only numpy).
- ``got_tools(graph)`` returns Arcane ``ToolSpec``s: map ``schema_dict()`` onto any LLM tool-calling
  API, or hand them to ``Arc1Agent`` (the bundled arc1-tiny is not trained on them; fine-tune first).
- ``GraphOfThought(graph, llm).reason(question)`` runs Graph-of-Thoughts (Besta et al., 2023).

Saved graphs (``to_json``) use the Node package's v2 format, so either side can load them.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .tools import ToolParam, ToolSpec

_STOPWORDS = set(
    "a an and are as at be been but by can could did do does doing for from had has have how i if in into is it its "
    "me my no not of on or our should so than that the their them then there these they this those to too was we "
    "were what when where which while who whom why will with would you your about after all also any before between "
    "both each few more most other over same some such only own under until up very just tell show give get use using".split()
)
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ed")
_WORD = re.compile(r"[^\w]+|_+")
_ATX = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

PARENT_CHILD, REFERENCE, NEXT = 0.8, 0.75, 0.4
UPWARD_FACTOR = 0.5  # walking child -> parent is weaker than going down
TITLE_BOOST = 3
TOP_TERMS = 12
BM25_K1, BM25_B = 1.2, 0.75


# ----------------------------------------------------------------------------- text

def _words(text: str) -> List[str]:
    return [w for w in _WORD.split(text.lower()) if w]


# ponytail: suffix stripping, not Porter; swap in a real stemmer if recall on inflections matters
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def tokenize(text: str) -> List[str]:
    """Lowercase stemmed word tokens without stopwords."""
    return [_stem(w) for w in _words(text) if len(w) > 1 and w not in _STOPWORDS]


def parse_sections(text: str) -> List[tuple]:
    """Split markdown into ``(title, level, body)`` by ATX/setext headings; level 0 is the preamble.

    ``#`` lines inside code fences are not headings.
    """
    # ponytail: markdown headings only; plain text lands in the preamble and is chunked under the root
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections, title, level, body, fence, i = [], "", 0, [], None, 0

    def flush():
        joined = "\n".join(body).strip()
        if title or joined:
            sections.append((title, level, joined))

    while i < len(lines):
        line = lines[i]
        fm = _FENCE.match(line)
        if fm and (fence is None or fm.group(1) == fence):
            fence = None if fence else fm.group(1)
        elif fence is None and line.strip():
            atx = _ATX.match(line)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            setext = (i == 0 or not lines[i - 1].strip()) and nxt and (
                set(nxt) == {"="} or (set(nxt) == {"-"} and len(nxt) >= 2 and not re.match(r"[-*+]\s", line.strip()))
            )
            if atx or setext:
                flush()
                title, level = (atx.group(2), len(atx.group(1))) if atx else (line.strip(), 1 if nxt[0] == "=" else 2)
                body = []
                i += 1 if atx else 2
                continue
        body.append(line)
        i += 1
    flush()
    return sections


def _chunk(text: str, max_chars: int) -> List[str]:
    """Pack paragraphs into chunks of at most ``max_chars``; over-long paragraphs are hard-split."""
    pieces = []
    for p in re.split(r"\n\s*\n", text):
        pieces.extend(p[k:k + max_chars] for k in range(0, max(len(p), 1), max_chars))
    chunks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    return chunks + [buf] if buf or not chunks else chunks


def _slug(text: str) -> str:
    return "-".join(_words(text))[:40].strip("-") or "doc"


# ----------------------------------------------------------------------------- graph

class DocumentGraph:
    """A dynamic multi-document knowledge graph with BM25 + bounded graph-expansion search.

    Nodes are document roots, heading sections and chunks of long sections. Edges are
    ``parent-child``, ``next`` (reading order), ``reference`` (text names another section's title)
    and ``semantic``. Re-adding a ``doc_id`` replaces that document; nothing else is touched.

    Args:
        embed: optional ``text -> vector`` callable, e.g. ``load_arc1().embed``. Semantic edges and
            extra query seeds then come from cosine similarity; otherwise from shared distinctive terms.
        min_similarity: link threshold (default 0.3 for embeddings, 0.08 for weighted term Jaccard).
    """

    def __init__(
        self,
        embed: Optional[Callable[[str], Sequence[float]]] = None,
        max_chunk_chars: int = 2000,
        semantic_neighbors: int = 5,
        min_similarity: Optional[float] = None,
    ):
        self.embed = embed
        self.max_chunk_chars = max_chunk_chars
        self.semantic_neighbors = semantic_neighbors
        self.min_similarity = min_similarity if min_similarity is not None else (0.3 if embed else 0.08)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._tf: Dict[str, Counter] = {}
        self._len: Dict[str, int] = {}
        self._postings: Dict[str, Dict[str, int]] = {}
        self._top: Dict[str, Dict[str, float]] = {}
        self._vec: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- mutation
    def add_document(self, content: str, title: str, doc_id: Optional[str] = None, description: str = "") -> Dict[str, Any]:
        """Index a document; re-using ``doc_id`` (default: slug of the title) replaces it."""
        doc_id = doc_id or _slug(title)
        self.remove_document(doc_id)
        sections = parse_sections(content)
        preamble = sections.pop(0)[2] if sections and sections[0][1] == 0 else ""
        if sections and sections[0][0].strip().lower() == title.strip().lower():
            preamble = "\n\n".join(filter(None, [preamble, sections.pop(0)[2]]))
        created: List[Dict[str, Any]] = []
        seq = count(1)

        def node(node_title, body, kind, level, parent):
            nid = doc_id if parent is None else f"{doc_id}#{next(seq)}"
            chunks = _chunk(body, self.max_chunk_chars)
            n = {"nodeId": nid, "docId": doc_id, "title": node_title, "content": chunks[0],
                 "type": kind, "level": level, "parentId": parent}
            self.nodes[nid], self.adj[nid] = n, {}
            created.append(n)
            if parent:
                self._link(parent, nid, PARENT_CHILD, "parent-child")
            prev = n
            for k, text in enumerate(chunks[1:], 2):
                part = node(f"{node_title} (part {k}/{len(chunks)})", text, "chunk", level + 1, nid)
                self._link(prev["nodeId"], part["nodeId"], NEXT, "next")
                prev = part
            return n

        root = node(title, preamble or description, "document", 0, None)
        stack, prev = [root], None
        for sec_title, level, body in sections:
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            n = node(sec_title, body, "section", level, stack[-1]["nodeId"])
            if prev:
                self._link(prev["nodeId"], n["nodeId"], NEXT, "next")
            stack.append(n)
            prev = n

        for n in created:
            self._index(n)
        self._link_references(created)
        for n in created:
            self._link_semantic(n["nodeId"])

        doc = {"docId": doc_id, "title": title, "description": description, "rootId": root["nodeId"],
               "nodeIds": [n["nodeId"] for n in created], "addedAt": datetime.now(timezone.utc).isoformat()}
        self.docs[doc_id] = doc
        return doc

    def remove_document(self, doc_id: str) -> bool:
        doc = self.docs.pop(doc_id, None)
        if not doc:
            return False
        for nid in doc["nodeIds"]:
            for term in self._tf.pop(nid, ()):
                postings = self._postings[term]
                postings.pop(nid, None)
                if not postings:
                    del self._postings[term]
            for other in self.adj.pop(nid, {}):
                self.adj.get(other, {}).pop(nid, None)
            for store in (self._len, self._top, self._vec, self.nodes):
                store.pop(nid, None)
        return True

    # ------------------------------------------------------------- queries
    def children(self, node_id: str) -> List[Dict[str, Any]]:
        return [self.nodes[o] for o, e in self.adj[node_id].items() if e["type"] == "parent-child" and e["from"] == node_id]

    def neighbors(self, node_id: str) -> List[tuple]:
        """``(node, edge)`` pairs for every edge touching ``node_id``."""
        return [(self.nodes[o], e) for o, e in self.adj[node_id].items()]

    def outline(self, doc_id: str, max_depth: int = 6) -> str:
        """Indented table of contents with node ids."""
        lines = []

        def walk(nid, depth):
            lines.append(f"{'  ' * depth}- [{nid}] {self.nodes[nid]['title']}")
            if depth < max_depth:
                for child in self.children(nid):
                    walk(child["nodeId"], depth + 1)

        walk(self.docs[doc_id]["rootId"], 0)
        return "\n".join(lines)

    def search(
        self,
        query: str,
        max_results: int = 8,
        max_seeds: int = 5,
        max_hops: int = 2,
        damping: float = 0.85,
        min_score: float = 0.1,
        max_activated: int = 200,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 (+ embedding) seeds, then a bounded multi-source Dijkstra over ``-log(edge weight)``.

        Each hit carries its ``score``, ``hops`` from the nearest seed, and ``via``/``edgeType``/``path``
        explaining how it was reached. Cost depends on the activated subgraph, not the corpus size.
        """
        allowed = set(doc_ids) if doc_ids else None
        lexical = self._bm25(query, allowed)
        if self.embed and self._vec:
            ids = [n for n in self._vec if not allowed or self.nodes[n]["docId"] in allowed]
            if ids:
                sims = np.stack([self._vec[n] for n in ids]) @ self._unit(query)
                for k in np.argsort(-sims)[:max_seeds]:
                    if sims[k] >= self.min_similarity:
                        lexical[ids[k]] = max(lexical.get(ids[k], 0.0), float(sims[k]))
        if not lexical:
            return []

        seeds = sorted(lexical.items(), key=lambda kv: -kv[1])[:max_seeds]
        reached: Dict[str, tuple] = {}  # nid -> (score, hops, via, edge type)
        best = dict(seeds)
        tie = count()
        heap = [(-s, next(tie), nid, 0, None, None) for nid, s in seeds]
        heapq.heapify(heap)
        while heap and len(reached) < max_activated:
            neg, _, nid, hops, via, etype = heapq.heappop(heap)
            if nid in reached:
                continue
            reached[nid] = (-neg, hops, via, etype)
            if hops >= max_hops:
                continue
            for other, edge in self.adj[nid].items():
                if other in reached or (allowed and self.nodes[other]["docId"] not in allowed):
                    continue
                upward = edge["type"] == "parent-child" and edge["to"] == nid
                score = -neg * edge["weight"] * (UPWARD_FACTOR if upward else 1.0) * damping
                if score >= min_score and score > best.get(other, 0.0):
                    best[other] = score
                    heapq.heappush(heap, (-score, next(tie), other, hops + 1, nid, edge["type"]))

        scored = []
        for nid in set(reached) | set(lexical):
            if not self.nodes[nid]["content"].strip():
                continue
            graph_score, _, via, _ = reached.get(nid, (0.0, 0, None, None))
            lex = lexical.get(nid, 0.0)
            # Noisy-OR: matched directly and reached through the graph beats either alone
            scored.append((1 - (1 - graph_score) * (1 - lex) if via else max(graph_score, lex), nid))
        scored.sort(key=lambda x: -x[0])

        hits = []
        for score, nid in scored[:max_results]:
            _, hops, via, etype = reached.get(nid, (0.0, 0, None, None))
            path = [nid]
            while reached.get(path[0], (0, 0, None))[2] and len(path) < 64:
                path.insert(0, reached[path[0]][2])
            n = self.nodes[nid]
            hits.append({"nodeId": nid, "docId": n["docId"], "title": n["title"], "content": n["content"],
                         "score": score, "lexicalScore": lexical.get(nid, 0.0), "hops": hops,
                         "via": via, "edgeType": etype, "path": path})
        return hits

    def retrieve(self, query: str, **search_kwargs) -> str:
        """Prompt-ready context: ``### Title [node-id]`` blocks for the search hits."""
        return "\n\n".join(f"### {h['title']} [{h['nodeId']}]\n{h['content']}" for h in self.search(query, **search_kwargs))

    # ------------------------------------------------------------- serialization
    def to_json(self) -> Dict[str, Any]:
        edges = list({id(e): e for links in self.adj.values() for e in links.values()}.values())
        return {"format": "graph-of-thought", "version": 2, "documents": list(self.docs.values()),
                "nodes": list(self.nodes.values()), "edges": edges}

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> "DocumentGraph":
        if data.get("format") != "graph-of-thought" or data.get("version") != 2:
            raise ValueError("Not a graph-of-thought v2 index")
        graph = cls(**kwargs)
        for n in data["nodes"]:
            graph.nodes[n["nodeId"]], graph.adj[n["nodeId"]] = dict(n), {}
        graph.docs = {d["docId"]: dict(d) for d in data["documents"]}
        for e in data["edges"]:
            graph._link(e["from"], e["to"], e["weight"], e["type"])
        for n in graph.nodes.values():
            graph._index(n)
        for nid in graph.nodes:
            graph._top[nid] = graph._top_terms(nid)
            if graph.embed:
                graph._vec[nid] = graph._unit(graph._text(nid))
        return graph

    # ------------------------------------------------------------- internals
    def _link(self, a: str, b: str, weight: float, kind: str) -> None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        existing = self.adj[a].get(b)
        if existing and existing["weight"] >= weight:
            return
        # Structural edges keep their direction; a stronger edge of another type replaces a weaker one
        edge = dict(existing, weight=weight) if existing and existing["type"] == "parent-child" \
            else {"from": a, "to": b, "weight": weight, "type": kind}
        self.adj[a][b] = self.adj[b][a] = edge

    def _index(self, n: Dict[str, Any]) -> None:
        tf = Counter(tokenize(n["content"]))
        for t in tokenize(n["title"]):
            tf[t] += TITLE_BOOST
        nid = n["nodeId"]
        self._tf[nid], self._len[nid] = tf, sum(tf.values())
        for term, c in tf.items():
            self._postings.setdefault(term, {})[nid] = c

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        return math.log(1 + (len(self.nodes) - df + 0.5) / (df + 0.5))

    # ponytail: exhaustive BM25 over the query terms' posting lists; add WAND top-k pruning past ~100k nodes
    def _bm25(self, query: str, allowed) -> Dict[str, float]:
        avg = sum(self._len.values()) / max(1, len(self._len))
        scores: Dict[str, float] = {}
        for term in set(tokenize(query)):
            idf = self._idf(term)
            for nid, tf in self._postings.get(term, {}).items():
                if allowed and self.nodes[nid]["docId"] not in allowed:
                    continue
                norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self._len[nid] / avg)
                scores[nid] = scores.get(nid, 0.0) + idf * tf * (BM25_K1 + 1) / norm
        top = max(scores.values(), default=1.0)
        return {k: v / top for k, v in scores.items()}

    def _link_references(self, created: List[Dict[str, Any]]) -> None:
        """Link a node to any section of the same document whose title its text names."""
        # ponytail: O(sections^2) per document; index titles by first word if documents get huge
        titles = [(f" {' '.join(_words(n['title']))} ", n) for n in created
                  if n["type"] == "section" and len(n["title"]) >= 4 and tokenize(n["title"])]
        for n in created:
            text = f" {' '.join(_words(n['content']))} "
            for phrase, target in titles:
                if target is not n and target["nodeId"] != n["parentId"] and phrase in text:
                    self._link(n["nodeId"], target["nodeId"], REFERENCE, "reference")

    def _text(self, nid: str) -> str:
        n = self.nodes[nid]
        return f"{n['title']}: {n['content']}"

    def _unit(self, text: str) -> np.ndarray:
        v = np.asarray(self.embed(text), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    # ponytail: top terms use the IDF at insert time and drift as the corpus grows; re-add a document to refresh
    def _top_terms(self, nid: str) -> Dict[str, float]:
        ranked = sorted(((t, (1 + math.log(c)) * self._idf(t)) for t, c in self._tf[nid].items()), key=lambda x: -x[1])
        return {t: self._idf(t) for t, _ in ranked[:TOP_TERMS]}

    # ponytail: linear scan over all nodes per insert; add an ANN / top-term inverted index past ~50k nodes
    def _link_semantic(self, nid: str) -> None:
        if self.semantic_neighbors <= 0:
            return
        sims: Dict[str, float] = {}
        if self.embed:
            self._vec[nid] = mine = self._unit(self._text(nid))
            for other, v in self._vec.items():
                if other != nid:
                    sims[other] = float(v @ mine)
        else:
            self._top[nid] = mine = self._top_terms(nid)
            total = sum(mine.values())
            for other, theirs in self._top.items():
                inter = sum(w for t, w in mine.items() if t in theirs)
                if other != nid and inter:
                    sims[other] = inter / (total + sum(theirs.values()) - inter)  # weighted Jaccard
        ranked = sorted((kv for kv in sims.items() if kv[1] >= self.min_similarity), key=lambda kv: -kv[1])
        for other, sim in ranked[: self.semantic_neighbors]:
            self._link(nid, other, min(0.9, 0.3 + 0.6 * sim), "semantic")


# ----------------------------------------------------------------------------- reasoning

def _loose_json(text: str) -> Dict[str, Any]:
    m = _JSON_OBJECT.search(text or "")
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    return data if isinstance(data, dict) else {}


class _Run:
    """State of one ``reason()`` call: the thought graph, its frontier and the evidence seen."""

    def __init__(self, got: "GraphOfThought", question: str):
        self.got, self.question = got, question
        self.thoughts: Dict[str, Dict[str, Any]] = {}
        self.frontier: List[Dict[str, Any]] = []
        self.calls = 0

    def add(self, content: str, operation: str, parents: List[Dict[str, Any]], evidence: List[str], **meta) -> Dict[str, Any]:
        t = {"id": f"t{len(self.thoughts) + 1}", "content": content, "score": None, "operation": operation,
             "parents": [p["id"] for p in parents], "evidence": evidence, **meta}
        self.thoughts[t["id"]] = t
        return t

    def ask(self, prompt: str) -> Optional[str]:
        if self.calls >= self.got.max_llm_calls:
            return None
        self.calls += 1
        return self.got.llm(prompt)

    def retrieve(self, query: str) -> List[str]:
        return [h["nodeId"] for h in self.got.graph.search(query, **self.got.search_kwargs)]

    def derive(self, operation: str, parents: List[Dict[str, Any]], cand: Dict[str, Any]) -> Dict[str, Any]:
        """Thought from an LLM candidate; a non-empty ``missing`` runs a follow-up graph search."""
        evidence = _union(*(p["evidence"] for p in parents))
        cited = [i for i in cand.get("evidence") or [] if isinstance(i, str) and i in self.got.graph.nodes]
        evidence = _union(cited, evidence)
        follow_up = str(cand.get("missing") or "").strip()
        if follow_up:
            evidence = _union(evidence, self.retrieve(follow_up))
        return self.add(str(cand["answer"]), operation, parents, evidence, followUp=follow_up)

    def header(self, evidence: List[str]) -> str:
        blocks = []
        for nid in evidence[: self.got.max_evidence]:
            n = self.got.graph.nodes.get(nid)
            if n:
                text = n["content"][: self.got.max_evidence_chars]
                blocks.append(f"[{nid}] {n['title']}\n{text}")
        return f"Question: {self.question}\n\nEvidence:\n" + ("\n\n".join(blocks) or "(no evidence found)")


def _union(*lists: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(i for lst in lists for i in lst))


_RULES = ('Answer only from the evidence. Cite the node ids you used in "evidence". If something needed is not in '
          'the evidence, put a short search query for it in "missing" (otherwise "").')
_ANSWER_JSON = 'Reply with JSON only:\n{"answer": "...", "evidence": ["node-id"], "missing": ""}'


class ops:
    """Graph-of-Thoughts operations. A plan is a list of them, applied in order to the frontier."""

    @staticmethod
    def retrieve(query: Optional[str] = None):
        """Search the graph (for the question, or a sub-query) and attach the hits to the frontier."""
        def run(r: _Run):
            ids = r.retrieve(query or r.question)
            if not r.frontier:
                r.frontier = [r.add(f"Evidence for: {r.question}", "retrieve", [], ids)]
            for t in r.frontier:
                t["evidence"] = _union(t["evidence"], ids)
        return run

    @staticmethod
    def generate(k: int = 3):
        """Branch every frontier thought into ``k`` candidate answers."""
        def run(r: _Run):
            nxt = []
            for parent in r.frontier:
                base = "" if parent["operation"] == "retrieve" else f"\n\nBuild on this partial answer:\n{parent['content']}"
                reply = r.ask(f"{r.header(parent['evidence'])}{base}\n\nPropose {k} distinct candidate answers "
                              f"(different readings of the question, evidence or lines of reasoning). {_RULES}\n\n"
                              'Reply with JSON only:\n{"thoughts": [{"answer": "...", "evidence": ["node-id"], "missing": ""}]}')
                if reply is None:
                    nxt.append(parent)
                    continue
                cands = _loose_json(reply).get("thoughts")
                cands = cands[:k] if isinstance(cands, list) else [{"answer": reply.strip()}]
                nxt.extend(r.derive("generate", [parent], c) for c in cands if isinstance(c, dict) and c.get("answer"))
            r.frontier = nxt or r.frontier
        return run

    @staticmethod
    def score():
        """Rate each frontier thought 0-10 against its evidence (stored as 0-1) with a critique."""
        def run(r: _Run):
            for t in r.frontier:
                if t["operation"] == "retrieve":
                    continue
                reply = r.ask(f"{r.header(t['evidence'])}\n\nCandidate answer:\n{t['content']}\n\nRate the candidate "
                              "from 0 to 10 for correctness, completeness and grounding in the evidence (unsupported "
                              'claims lower the score).\n\nReply with JSON only:\n{"score": 0, "critique": "..."}')
                if reply is None:
                    continue
                data = _loose_json(reply)
                raw = data.get("score")
                if raw is None:
                    m = re.search(r"\d+(?:\.\d+)?", reply)
                    raw = m.group(0) if m else None
                try:
                    t["score"] = max(0.0, min(1.0, float(raw) / 10))
                except (TypeError, ValueError):
                    pass
                if data.get("critique"):
                    t["critique"] = data["critique"]
        return run

    @staticmethod
    def keep_best(n: int = 1):
        """Prune the frontier to the ``n`` highest-scoring thoughts."""
        def run(r: _Run):
            r.frontier = sorted(r.frontier, key=lambda t: -(t["score"] or 0))[:n]
        return run

    @staticmethod
    def aggregate():
        """Merge the frontier into one thought (the step a tree of thoughts can't do)."""
        def run(r: _Run):
            if len(r.frontier) < 2:
                return
            listing = "\n\n".join(
                f"Candidate {i}" + (f" (score {round(t['score'] * 10)}/10)" if t["score"] is not None else "") + f":\n{t['content']}"
                for i, t in enumerate(r.frontier, 1))
            reply = r.ask(f"{r.header(_union(*(t['evidence'] for t in r.frontier)))}\n\n{listing}\n\nMerge the candidates "
                          "into one answer: keep every supported point, resolve conflicts using the evidence, drop "
                          f"unsupported claims. {_RULES}\n\n{_ANSWER_JSON}")
            if reply is not None:
                data = _loose_json(reply)
                r.frontier = [r.derive("aggregate", r.frontier, data if data.get("answer") else {"answer": reply.strip()})]
        return run

    @staticmethod
    def refine(max_rounds: int = 2):
        """Improve each thought; repeats while the model asks for (and gets) more evidence."""
        def run(r: _Run):
            out = []
            for cur in r.frontier:
                for _ in range(max_rounds):
                    critique = f"\n\nCritique:\n{cur['critique']}" if cur.get("critique") else ""
                    reply = r.ask(f"{r.header(cur['evidence'])}\n\nCurrent answer:\n{cur['content']}{critique}\n\n"
                                  "Improve the answer: fix errors, fill gaps from the evidence, remove unsupported "
                                  f"claims. {_RULES}\n\n{_ANSWER_JSON}")
                    if reply is None:
                        break
                    data = _loose_json(reply)
                    cur = r.derive("refine", [cur], data if data.get("answer") else {"answer": reply.strip()})
                    if not cur["followUp"]:
                        break
                out.append(cur)
            r.frontier = out
        return run


def default_plan() -> List[Callable[[_Run], None]]:
    """retrieve -> generate(3) -> score -> keep_best(2) -> aggregate -> refine -> score (about 7 LLM calls)."""
    return [ops.retrieve(), ops.generate(3), ops.score(), ops.keep_best(2), ops.aggregate(), ops.refine(), ops.score()]


class GraphOfThought:
    """Graph-of-Thoughts reasoning (Besta et al., 2023) grounded in a ``DocumentGraph``.

    Thoughts are vertices and their ``parents`` are the edges. Any generate/aggregate/refine step may
    report ``missing`` information, which triggers a new graph search mid-reasoning.

    Args:
        graph: the ``DocumentGraph`` to search.
        llm: any ``prompt -> text`` callable (Claude, OpenAI, Ollama, a local model...).
        search_kwargs: forwarded to ``DocumentGraph.search``.
    """

    def __init__(self, graph: DocumentGraph, llm: Callable[[str], str], max_llm_calls: int = 24,
                 max_evidence: int = 10, max_evidence_chars: int = 1500, **search_kwargs):
        self.graph, self.llm = graph, llm
        self.max_llm_calls = max_llm_calls
        self.max_evidence, self.max_evidence_chars = max_evidence, max_evidence_chars
        self.search_kwargs = search_kwargs

    # ponytail: LLM calls run sequentially; fan out generate/score with a ThreadPoolExecutor if latency matters
    def reason(self, question: str, plan: Optional[Sequence[Callable[[_Run], None]]] = None) -> Dict[str, Any]:
        """Answer ``question``. Returns ``answer``, ``citations``, the whole ``thoughts`` graph and ``llm_calls``."""
        r = _Run(self, question)
        for op in plan or default_plan():
            op(r)
        answers = [t for t in r.frontier if t["operation"] != "retrieve"]
        best = max(answers, key=lambda t: t["score"] if t["score"] is not None else -1, default=None)
        citations = [{"nodeId": i, "docId": self.graph.nodes[i]["docId"], "title": self.graph.nodes[i]["title"]}
                     for i in (best["evidence"] if best else []) if i in self.graph.nodes]
        return {"question": question, "answer": best["content"] if best else "", "best": best,
                "thoughts": list(r.thoughts.values()), "citations": citations, "llm_calls": r.calls}


# ----------------------------------------------------------------------------- harness

def got_tools(graph: DocumentGraph, reasoner: Optional[GraphOfThought] = None, writable: bool = False) -> List[ToolSpec]:
    """Agent tools over ``graph`` as ``ToolSpec``s.

    Map ``spec.schema_dict()`` onto any LLM tool-calling API and run ``execute_tools(calls, specs)``.
    Read-only by default; ``writable=True`` adds ``got_add_document`` / ``got_remove_document``.
    ``got_reason`` is included when a ``reasoner`` is given.
    """
    def p(name, description):
        return ToolParam(name=name, type="string", description=description)

    def search(query):
        return [{"nodeId": h["nodeId"], "title": h["title"], "score": round(h["score"], 3),
                 "reachedVia": f"{h['edgeType']} from {h['via']}" if h["via"] else "direct match",
                 "snippet": h["content"][:300]} for h in graph.search(query)]

    def read_node(node_id):
        if node_id not in graph.nodes:
            raise KeyError(f"Unknown node: {node_id}. Use got_search or got_outline to find node ids.")
        n = graph.nodes[node_id]
        links = [{"nodeId": o["nodeId"], "title": o["title"], "type": e["type"]}
                 for o, e in graph.neighbors(node_id) if e["type"] != "parent-child"]
        return {"nodeId": node_id, "title": n["title"], "content": n["content"], "parentId": n["parentId"],
                "children": [c["nodeId"] for c in graph.children(node_id)], "links": links}

    tools = [
        ToolSpec("got_search", "Search the document knowledge graph for sections about a topic",
                 [p("query", "what to look for")], search),
        ToolSpec("got_read_node", "Read the full text of a section by its node id, with its linked sections",
                 [p("node_id", "node id such as user-guide#3")], read_node),
        ToolSpec("got_outline", "Show the table of contents of a document",
                 [p("doc_id", "document id")], graph.outline),
        ToolSpec("got_list_documents", "List the documents in the knowledge graph", [],
                 lambda: [{"docId": d["docId"], "title": d["title"], "nodes": len(d["nodeIds"])} for d in graph.docs.values()]),
    ]
    if writable:
        tools += [
            ToolSpec("got_add_document", "Add or replace a document in the knowledge graph",
                     [p("content", "document text; markdown headings become sections"), p("title", "document title")],
                     lambda content, title: graph.add_document(content, title)["docId"]),
            ToolSpec("got_remove_document", "Remove a document from the knowledge graph",
                     [p("doc_id", "document id")], graph.remove_document),
        ]
    if reasoner:
        tools.append(ToolSpec("got_reason", "Answer a question from the documents with cited sections",
                              [p("question", "the question to answer")],
                              lambda question: {k: v for k, v in reasoner.reason(question).items() if k in ("answer", "citations")}))
    return tools
