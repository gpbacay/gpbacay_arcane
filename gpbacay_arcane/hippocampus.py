"""Hippocampus: a fast-learning memory of decided examples for System 1 models.

Named for its role in the brain (complementary learning systems): the neocortex, here the model, learns
slowly and holds general knowledge in its weights; the hippocampus stores single episodes at once and
biases the cortex's decision when a similar cue comes back. Examples are stored by ``remember``, recalled
by similarity, and mixed with the model's own judgment, never written into its weights.

RAG gives an LLM knowledge its weights lack by retrieving text at request time. A System 1 decision
model (ARC 1, a small classifier, an LLM used as a router) does not read retrieved prose, so Hippocampus
retrieves *decisions* instead: ``remember(text, label)`` stores a decided example in a ``DocumentGraph``,
and at request time the examples most like the input vote for their labels (kNN-LM style, weighted by
search score). The vote is combined with the model's own probabilities. New examples take effect on the
next request, with no fine-tuning; a label with no examples falls back to the model alone.

Tool examples can also carry their arguments (``remember(text, tool, arguments)``). A remembered request
becomes a pattern ("rate {title} {stars} stars"), and when it matches a new request the arguments are
copied from the new request's words, so a tool the model was never trained on gets its arguments from
a few examples instead of from fine-tuning.

Model-agnostic: ``model`` is anything below, or ``None`` for memory-only decisions.

- a callable ``(text, labels, **kwargs) -> {label: probability}``;
- an object with ``classify(text, labels, **kwargs)`` returning a dict with ``"distribution"``
  (``Arc1Agent``, or a wrapper around any classifier or LLM);
- for tool calling, an object with ``run(prompt, tools=, tool_prior=, execute=, **kwargs)`` returning
  ``function_calls``, that combines ``tool_prior`` (tool name -> probability the tool applies) with its
  own firing decision (``Arc1Agent``).

Learning without a person in the loop: give ``teacher`` a stronger, slower model (an LLM, a bigger classifier).
When ``classify`` is not confident it asks the teacher, returns the answer and remembers it, so the next
similar request is decided here, fast, without the teacher. Without any outside signal a memory can only echo
its model; the teacher is that signal. ``observe`` takes outcomes from anywhere else (a rule, a tool result).

Better recall: ``Hippocampus(model, occipital=Occipital())`` also recalls examples by their letters (see
``occipital.py``), so a typo or another word form still finds its neighbors. It changes which examples vote, not
the model, so it helps whatever model is behind the memory.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from .got import DocumentGraph
from .tools import _WORD_NUMBERS, coerce_value, execute_tools, validate_calls_against_tools

MEMORY_PREFIX = "memory-"
AGREE = 3  # neighbors that must back a label before the memory alone counts as sure (see _decisive)
# ponytail: English clause boundaries; a value containing "and" ("Tom and Jerry") is cut where a pattern has an open end
_JOINER = r"(?:and|then|plus|also)\b"
_CLAUSE_END = rf"(?=\s*(?:$|[.,;!?\n]|{_JOINER}))"
_CLAUSE_START = rf"(?:^|[.,;!?\n]|\b{_JOINER})\W*"
_OPEN_VALUE = rf"(?=\w)(?:(?!\b{_JOINER})[^.,;!?\n])+?"  # a value at an open end holds no joiner
_NUMBER = r"-?\d+(?:[.,]\d+)*|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))


def _doc_id(text: str) -> str:
    return MEMORY_PREFIX + hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()[:12]


def _template(text: str, arguments: Dict[str, Any], params: Dict[str, Any]):
    """``(regex, slot keys, fixed arguments)`` for a remembered request, or ``None`` if it has no words.

    Each argument value found in ``text`` becomes a slot typed by its parameter; values not in the text
    (booleans, presets) are fixed arguments implied by the pattern's literal words ("switch bluetooth off").
    """
    spans, fixed = [], {}
    for key, value in arguments.items():
        if key not in params:
            continue
        m = None if isinstance(value, bool) else re.search(rf"(?<!\w){re.escape(str(value))}(?!\w)", text, re.IGNORECASE)
        if m and not any(m.start() < e and s < m.end() for s, e, _ in spans):
            spans.append((m.start(), m.end(), key))
        else:
            fixed[key] = value
    items, keys, pos = [], [], 0
    for s, e, key in sorted(spans):
        items += [re.escape(w) for w in re.findall(r"\w+", text[pos:s])] + [None]
        keys.append(key)
        pos = e
    items += [re.escape(w) for w in re.findall(r"\w+", text[pos:])]
    if not items:
        return None
    parts, slots = [], iter(keys)
    for n, item in enumerate(items):
        if item is not None:
            parts.append(item)
            continue
        param = params[next(slots)]
        if param.enum:
            body = "|".join(re.escape(str(o)) for o in sorted(param.enum, key=len, reverse=True))
        elif (param.type or "string").lower() in ("integer", "int", "number", "float"):
            body = _NUMBER
        elif n == 0 or n == len(items) - 1:
            body = _OPEN_VALUE + (_CLAUSE_END if n == len(items) - 1 else "")
        else:
            body = r"[^.,;!?\n]+?"
        parts.append(f"({body})")
    head = _CLAUSE_START if items[0] is None else r"\b"
    tail = r"\b" if items[-1] is not None else ""
    return re.compile(head + r"\W+".join(parts) + tail, re.IGNORECASE), keys, fixed


class Hippocampus:
    """A System 1 model plus a dynamic memory of decided examples.

    Examples are documents whose id starts with ``memory-``, with the label in ``description`` and any
    tool arguments under ``arguments``, so ``graph.to_json()`` / ``DocumentGraph.from_json`` persist the
    memory and a graph shared with other documents keeps them apart. The default graph has no semantic
    links: on CLINC150 they lowered accuracy and made every insert scan the whole memory.

    Args:
        model: see the module docstring; ``None`` decides from memory alone.
        graph: where examples live; a new lexical ``DocumentGraph`` by default.
        k: neighbors that vote.
        weight: share of a ``classify`` distribution given to the vote when any neighbor is found.
        max_size: cap on remembered examples; ``observe`` evicts its own oldest writes past it
            (``remember``ed examples are never evicted). ``None`` = unbounded.
        confident: ``observe`` skips an example the memory and model already decide this confidently.
        threshold: ``observe`` commits a candidate once its accumulated trust reaches ``threshold``.
        leak: share of every pending candidate's evidence lost per ``observe`` call (leaky integrate-and-fire).
            ``0`` keeps evidence until it is cancelled or pushed out; set it to about 1 / the calls between
            repeats of a text to make old evidence count for less.
        max_pending: candidates kept while they gather evidence; the stalest is dropped past it.
        teacher: ``(text, labels) -> label | {label: probability} | None``, a stronger model that ``classify``
            asks when its own decision is under ``confident`` (see the module docstring). An answer under
            ``confident`` is not trusted. ``None`` (default) asks nobody.
        occipital: an ``Occipital`` of its own, which also recalls examples by their letters. ``None`` (default)
            searches words only.
    """

    def __init__(self, model: Any = None, graph: Optional[DocumentGraph] = None, k: int = 8, weight: float = 0.5,
                 max_size: Optional[int] = None, confident: float = 0.8, threshold: float = 1.0, leak: float = 0.0,
                 max_pending: int = 10000, teacher: Any = None, occipital: Any = None):
        self.model, self.k, self.weight, self.teacher, self.occipital = model, k, weight, teacher, occipital
        self.max_size, self.confident, self.threshold, self.leak = max_size, confident, threshold, leak
        self.max_pending = max_pending
        self.graph = graph if graph is not None else DocumentGraph(semantic_neighbors=0)
        self._label: Dict[str, Optional[str]] = {}
        self._by_label: Dict[Optional[str], Set[str]] = {}
        self._pending: Dict[str, Dict[Optional[str], float]] = {}  # doc_id -> label -> potential
        for d, doc in self.graph.docs.items():  # a loaded graph brings its memory with it
            if d.startswith(MEMORY_PREFIX):
                self._index(d, doc["description"] or None)
                if occipital is not None:
                    occipital.add(d, "\n".join(self.graph.nodes[n]["content"] for n in doc["nodeIds"]))

    # ------------------------------------------------------------- memory
    def remember(self, text: str, label: Optional[str], arguments: Optional[Dict[str, Any]] = None) -> str:
        """Store ``text`` decided as ``label`` (a class label or a tool name; ``None`` = no tool applies).

        ``arguments`` (for a tool) are that call's arguments; values that appear in ``text`` teach where
        each argument sits in a request. Remembering the same text again replaces it, so corrections
        overwrite mistakes.
        """
        return self._store(text, label, arguments, "manual")

    def _store(self, text: str, label: Optional[str], arguments: Optional[Dict[str, Any]], source: str) -> str:
        if not text.strip():
            raise ValueError("remember() needs non-empty text")  # an empty node would index the label instead
        doc_id = _doc_id(text)
        self.forget(text)
        self.graph.add_document(text, "", doc_id=doc_id, description=label or "")
        doc = self.graph.docs[doc_id]
        doc["source"] = source  # saved by to_json with the document
        if arguments:
            doc["arguments"] = dict(arguments)
        self._index(doc_id, label or None)
        if self.occipital is not None:
            self.occipital.add(doc_id, text)
        return doc_id

    def observe(self, text: str, label: Optional[str], arguments: Optional[Dict[str, Any]] = None,
                trust: float = 1.0, labels: Optional[Sequence[str]] = None, **model_kwargs) -> Optional[str]:
        """Learn from an outcome: ``label`` is what turned out right for ``text`` (a user correction, a tool
        that ran, a ground truth). Returns the new example's id, or ``None`` when nothing was stored.

        Stores only what the memory (with its model) does not already decide confidently, so it grows
        where the model is weak. ``labels`` is the decision space that judgment is made over (default: every
        label in the memory); give it when the memory holds examples for other decisions. ``trust`` (0..1+)
        is the outcome's reliability; evidence for the same text and label accumulates, decays by ``leak``
        per call, is cancelled by evidence for another label, and commits at ``threshold``, so a noisy
        signal must repeat before it becomes memory. The default ``trust=1.0`` commits at once. Past
        ``max_size`` the oldest ``observe``d example is forgotten. An example you ``remember``ed is never
        overwritten here (``None`` is returned); change it with ``remember`` or ``forget``. Extra keyword
        arguments go to the model, as in ``classify``.
        """
        existing = self.graph.docs.get(_doc_id(text))
        if existing and existing.get("source", "manual") == "manual":
            return None
        space = None if labels is None else [str(x) for x in dict.fromkeys(labels)]
        if space is None:
            space = [lab for lab in self._by_label if lab]
        if label and label not in space:
            space.append(label)
        if label is None or len(space) < 2:  # a model's choice among fewer than two labels says nothing
            seen = self.react(text, None if labels is None else space)["votes"].get(label or None, 0.0)
        else:
            seen = self._decide(text, space, **model_kwargs)["distribution"].get(label, 0.0)
        if self.leak:  # ponytail: O(pending) per call; only paid when a leak is asked for
            decayed = {d: {lab: v * (1 - self.leak) for lab, v in c.items() if v * (1 - self.leak) > 0.01}
                       for d, c in self._pending.items()}
            self._pending = {d: c for d, c in decayed.items() if c}
        if seen >= self.confident:
            return None
        doc_id = _doc_id(text)
        if doc_id not in self._pending and len(self._pending) >= self.max_pending:
            del self._pending[next(iter(self._pending))]  # the stalest candidate
        cands = self._pending.setdefault(doc_id, {})
        for lab in list(cands):
            if lab != (label or None):
                cands[lab] -= trust  # contradicting evidence
                if cands[lab] <= 0.01:
                    del cands[lab]
        cands[label or None] = cands.get(label or None, 0.0) + trust
        if cands[label or None] < self.threshold:
            return None
        self._store(text, label, arguments, "auto")  # also clears the pending candidates for this text
        self._evict(keep=doc_id)
        return doc_id

    def _evict(self, keep: str) -> None:
        """Drop the oldest ``observe``d examples past ``max_size``. Tried and dropped: scoring examples by how
        often a neighbor lookup agreed with the outcome; it did no better than FIFO or random eviction."""
        # ponytail: O(memory) scan per eviction; keep a queue of auto ids if max_size gets large
        while self.max_size and len(self._label) > self.max_size:
            victim = next((d for d in self._label if d != keep and self.graph.docs[d].get("source") == "auto"), None)
            if victim is None:
                return  # only manual examples left: they are never evicted
            self._by_label[self._label.pop(victim)].discard(victim)
            self.graph.remove_document(victim)
            if self.occipital is not None:
                self.occipital.remove(victim)

    def forget(self, text: str) -> bool:
        doc_id = _doc_id(text)
        self._pending.pop(doc_id, None)
        if doc_id in self._label:
            self._by_label[self._label.pop(doc_id)].discard(doc_id)
        if self.occipital is not None:
            self.occipital.remove(doc_id)
        return self.graph.remove_document(doc_id)

    def labels(self) -> Dict[str, Optional[str]]:
        """``doc_id -> label`` for every remembered example."""
        return dict(self._label)

    def _index(self, doc_id: str, label: Optional[str]) -> None:
        self._label[doc_id] = label
        self._by_label.setdefault(label, set()).add(doc_id)

    # ------------------------------------------------------------- retrieval
    def neighbors(self, text: str, labels: Optional[Sequence[Optional[str]]] = None) -> List[Dict[str, Any]]:
        """Remembered examples most like ``text`` (only those labeled one of ``labels``, if given)."""
        wanted = self._by_label.keys() if labels is None else set(labels) & self._by_label.keys()
        if not any(self._by_label[lab] for lab in wanted):
            return []
        # ponytail: the filter set is O(examples of the wanted labels); skipped when the graph is all memory
        allowed = None
        if len(self._label) < len(self.graph.docs) or wanted != self._by_label.keys():
            allowed = set().union(*(self._by_label[lab] for lab in wanted))
        if self.occipital is not None:
            return self._fused_neighbors(text, allowed)
        return [{"docId": h["docId"], "text": h["content"], "label": self._label[h["docId"]], "score": h["score"]}
                for h in self.graph.search(text, max_results=self.k, doc_ids=allowed)]

    def _fused_neighbors(self, text: str, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
        """Word scores (the graph's BM25, per example) fused with Occipital's letter scores. Skips the graph walk,
        which for a memory of one-node examples adds nothing to BM25."""
        lexical: Dict[str, float] = {}
        for nid, score in self.graph._bm25(text, allowed).items():
            d = self.graph.nodes[nid]["docId"]
            if d in self._label:
                lexical[d] = max(lexical.get(d, 0.0), score)
        fused = self.occipital.fuse(text, lexical, allowed.__contains__ if allowed is not None else None)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:self.k]
        return [{"docId": d, "text": self.graph.nodes[self.graph.docs[d]["rootId"]]["content"],
                 "label": self._label[d], "score": score,
                 "letters": d not in lexical}  # True: only the letters found it (a typo or another word form)
                for d, score in top]

    def react(self, text: str, labels: Optional[Sequence[Optional[str]]] = None) -> Dict[str, Any]:
        """The memory's own decision, no model involved: ``votes`` (label -> share of the neighbors'
        score, summing to 1) and the ``neighbors`` that cast them. Empty when nothing similar is stored."""
        hits = self.neighbors(text, labels)
        total = sum(h["score"] for h in hits)
        votes: Dict[Optional[str], float] = {}
        for h in hits:
            votes[h["label"]] = votes.get(h["label"], 0.0) + h["score"] / total
        return {"votes": votes, "neighbors": hits}

    def arguments(self, prompt: str, tool) -> Optional[Dict[str, Any]]:
        """Arguments for ``tool`` (a ``ToolSpec``) copied from ``prompt`` through the most similar
        remembered request whose pattern matches it; ``None`` when none matches."""
        found = self._match(prompt, tool)
        return found[0] if found else None

    def _match(self, prompt: str, tool) -> Optional[tuple]:
        """``(arguments, (start, end) of the matched words in prompt)`` or ``None``."""
        params = {p.name: p for p in tool.parameters}
        for hit in self.neighbors(prompt, [tool.name]):
            stored = self.graph.docs[hit["docId"]].get("arguments")
            if not stored:
                continue
            built = _template(self.graph.nodes[hit["docId"]]["content"], stored, params)
            m = built and built[0].search(prompt)
            if not m:
                continue
            keys, fixed = built[1], built[2]
            args = dict(fixed)
            for key, surface in zip(keys, m.groups()):
                ok, value = coerce_value(params[key].type, surface.strip())
                if not ok:
                    break
                args[key] = value
            else:
                return args, m.span()
        return None

    # ------------------------------------------------------------- decisions
    def _model_distribution(self, text: str, labels: List[str], **kwargs) -> tuple:
        if self.model is None:
            return {}, {}
        if hasattr(self.model, "classify"):
            out = dict(self.model.classify(text, labels, **kwargs))
            return dict(out["distribution"]), out
        return dict(self.model(text, labels, **kwargs)), {}

    def classify(self, text: str, labels: Sequence[str], **model_kwargs) -> Dict[str, Any]:
        """Pick one of ``labels``: the model's distribution mixed with the memory's vote.

        Returns the model's own output fields (if any), with ``label``, ``confidence``, ``distribution``,
        ``source`` (``model``, ``memory`` or ``model+memory``) and ``neighbors`` (the evidence). With no
        model and no similar example, ``label`` is ``None``: Hippocampus abstains rather than guess.

        With a ``teacher``, a decision under ``confident`` is put to it: a trusted answer becomes the result
        (``source="teacher"``) and is remembered as an ``observe``d example, so the next similar request is
        decided without it. If the teacher raises, the memory's own decision stands, with ``teacher_error``.
        A text you ``remember``ed is never put to the teacher.
        """
        labels = [str(x) for x in dict.fromkeys(labels)]
        out = self._decide(text, labels, **model_kwargs)
        doc_id = _doc_id(text)
        known = self.graph.docs.get(doc_id)
        if (self.teacher is None or out["confidence"] >= self.confident or self._decisive(out)
                or (known and known.get("source", "manual") == "manual")):
            return out
        try:
            taught = self._ask_teacher(text, labels)
        except Exception as exc:  # noqa: BLE001 - a down teacher must not take the memory down with it
            return {**out, "teacher_error": f"{type(exc).__name__}: {exc}"}
        if taught is None:
            return out
        label, dist = taught
        self._store(text, label, None, "auto")
        self._evict(keep=doc_id)
        return {**out, "label": label, "confidence": dist[label], "distribution": dist, "source": "teacher"}

    def _decisive(self, out: Dict[str, Any]) -> bool:
        """The memory alone is sure: at least ``AGREE`` neighbors back the decided label and hold ``confident`` of the
        vote. The blended confidence can't say so, since it is capped by the model's own, weak, probability."""
        hits = out["neighbors"]
        total = sum(h["score"] for h in hits)
        backing = [h["score"] for h in hits if h["label"] == out["label"]]
        return len(backing) >= AGREE and sum(backing) >= self.confident * total

    def _ask_teacher(self, text: str, labels: List[str]) -> Optional[tuple]:
        """``(label, distribution)`` from the teacher when it is confident about one of ``labels``, else ``None``."""
        answer = self.teacher(text, labels)
        if answer is None:
            return None
        dist = {str(answer): 1.0} if isinstance(answer, str) else {str(l): float(p) for l, p in dict(answer).items()}
        dist = {lab: dist.get(lab, 0.0) for lab in labels}
        total = sum(dist.values())
        if total <= 0:
            return None
        dist = {lab: p / total for lab, p in dist.items()}
        label = max(dist, key=dist.get)
        return (label, dist) if dist[label] >= self.confident else None

    def _decide(self, text: str, labels: List[str], **model_kwargs) -> Dict[str, Any]:
        model_dist, out = self._model_distribution(text, labels, **model_kwargs)
        memory = self.react(text, labels)
        votes = memory["votes"]
        if model_dist and votes:
            dist, source = {lab: (1 - self.weight) * model_dist.get(lab, 0.0) + self.weight * votes.get(lab, 0.0)
                            for lab in labels}, "model+memory"
        elif votes:
            dist, source = {lab: votes.get(lab, 0.0) for lab in labels}, "memory"
        else:
            dist, source = {lab: model_dist.get(lab, 0.0) for lab in labels}, "model"
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        best = ranked[0] if ranked and ranked[0][1] > 0 else (None, 0.0)
        out.update(label=best[0], confidence=best[1], distribution=dict(ranked), source=source,
                   neighbors=memory["neighbors"])
        return out

    def run(self, prompt: str, tools: Optional[Sequence[Any]] = None, execute: bool = True,
            model: Any = None, **model_kwargs) -> Dict[str, Any]:
        """Tool calling: memory votes fire tools and remembered requests fill their arguments.

        The votes go to ``model.run`` as ``tool_prior``. ``model`` defaults to this memory's model, so a
        caller such as ``System1`` can supply the predictor without storing it here. Every example votes,
        including ones for tools not offered and ``None`` (no tool), so a request that resembles something
        else does not fire an offered tool. Then, for each offered tool, a remembered request whose pattern
        matches supplies the arguments it covers (over the model's) and adds the call if the model held it
        back; ``confidence`` is then ``None``, since the model's calibrated probability no longer describes
        the call. Words a pattern matched belong to that call, so another model call whose argument copies
        them is dropped (ARC 1's own rule: a word belongs to one argument). ``memory_arguments`` names the
        tools whose arguments came from memory. For memory-only routing use ``react``.
        """
        model = self.model if model is None else model
        if not hasattr(model, "run"):
            raise TypeError("run() needs a model with run(prompt, tools=, tool_prior=); use react() for memory-only routing")
        memory = self.react(prompt)
        prior = {lab: p for lab, p in memory["votes"].items() if lab}
        out = dict(model.run(prompt, tools=tools, tool_prior=prior or None, execute=False, **model_kwargs))
        offered = list(tools if tools is not None else getattr(self.model, "tools", []))
        calls = {c["name"]: {"name": c["name"], "arguments": dict(c["arguments"])} for c in out["function_calls"]}
        filled, claimed = [], []
        for tool in offered:
            found = self._match(prompt, tool)
            if found is None:
                continue
            calls.setdefault(tool.name, {"name": tool.name, "arguments": {}})["arguments"].update(found[0])
            out["confidence"] = None  # the model's calibrated probability no longer describes this call
            filled.append(tool.name)
            claimed.append(found[1])

        def overlaps_claimed(value: Any) -> bool:
            i = prompt.lower().find(str(value).lower()) if isinstance(value, str) and value.strip() else -1
            return i >= 0 and any(i < e and s < i + len(value) for s, e in claimed)

        calls = {name: c for name, c in calls.items()
                 if name in filled or not any(overlaps_claimed(v) for v in c["arguments"].values())}
        out["function_calls"] = validate_calls_against_tools(list(calls.values()), offered)
        out["results"] = execute_tools(out["function_calls"], offered) if execute else []
        if filled:
            out["source"] = "model+memory"
        out["memory_arguments"], out["neighbors"] = filled, memory["neighbors"]
        return out
