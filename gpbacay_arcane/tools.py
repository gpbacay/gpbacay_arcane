"""Tool specs, validation, and the ARC 1 agent (tool calling, extraction, embeddings)."""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tokenization import BytePairTokenizer


@dataclass
class ToolParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: Optional[List[str]] = None
    # Optional plain-language hint per enum option ("billing": "charges, invoices, refunds").
    enum_descriptions: Optional[Dict[str, str]] = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: List[ToolParam] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None

    def schema_dict(self) -> Dict[str, Any]:
        props = {}
        required = []
        for p in self.parameters:
            entry: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                entry["enum"] = list(p.enum)
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[List[ToolParam]] = None,
):
    """Decorator that turns a Python function into a ``ToolSpec``."""

    def wrap(fn: Callable[..., Any]) -> ToolSpec:
        tool_name = name or fn.__name__
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        params = list(parameters or [])
        if not params:
            # Infer string params from annotations / defaults when possible.
            import inspect

            sig = inspect.signature(fn)
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                params.append(
                    ToolParam(
                        name=pname,
                        type="string",
                        description=pname,
                        required=param.default is inspect.Parameter.empty,
                    )
                )
        return ToolSpec(name=tool_name, description=desc, parameters=params, handler=fn)

    return wrap


def format_tools_prompt(user_message: str, tools: Sequence[ToolSpec], system: str = "") -> str:
    """Serialize tools + user turn into a next-token LM prompt."""
    catalog = [t.schema_dict() for t in tools]
    parts = []
    if system:
        parts.append(system.strip())
    parts.append("TOOLS:")
    parts.append(json.dumps(catalog, ensure_ascii=False))
    parts.append(
        "Respond with ONLY a JSON object of the form "
        '{"reasoning":"...","function_calls":[{"name":"...","arguments":{...}}]}. '
        "If no tool applies, return {\"reasoning\":\"...\",\"function_calls\":[]}."
    )
    parts.append("USER:")
    parts.append(user_message.strip())
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def format_extract_prompt(text: str, record_schema: Dict[str, Any]) -> str:
    """Extraction = one synthetic tool named extract_record."""
    extract_tool = ToolSpec(
        name="extract_record",
        description="Extract a typed record from the passage. Every value must be grounded in the text.",
        parameters=[
            ToolParam(name=k, type=str(v.get("type", "string")), description=str(v.get("description", k)))
            for k, v in record_schema.items()
        ],
    )
    return format_tools_prompt(
        f"Extract fields from this passage:\n{text}",
        [extract_tool],
        system="You extract structured JSON. Copy spans from the passage; do not invent values.",
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_agent_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of model output into the ARC 1 response shape."""
    text = (text or "").strip()
    if not text:
        return {"reasoning": "", "function_calls": []}
    candidates = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.insert(0, match.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        calls = data.get("function_calls", data.get("tool_calls", []))
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            continue
        normalized = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("function")
            args = call.get("arguments") or call.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if not name:
                continue
            normalized.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
        return {
            "reasoning": str(data.get("reasoning", "")),
            "function_calls": normalized,
        }
    return {"reasoning": text[:200], "function_calls": []}


def validate_calls_against_tools(
    calls: List[Dict[str, Any]],
    tools: Sequence[ToolSpec],
) -> List[Dict[str, Any]]:
    """Drop unknown tools and strip unknown / mistyped arguments."""
    by_name = {t.name: t for t in tools}
    valid = []
    for call in calls:
        spec = by_name.get(call.get("name", ""))
        if spec is None:
            continue
        args_in = call.get("arguments") or {}
        cleaned = {}
        allowed = {p.name: p for p in spec.parameters}
        for key, value in args_in.items():
            if key not in allowed:
                continue
            param = allowed[key]
            if param.enum and str(value) not in param.enum:
                continue
            cleaned[key] = value
        missing = [p.name for p in spec.parameters if p.required and p.name not in cleaned]
        if missing:
            continue
        valid.append({"name": spec.name, "arguments": cleaned})
    return valid


def execute_tools(
    calls: List[Dict[str, Any]],
    tools: Sequence[ToolSpec],
) -> List[Any]:
    by_name = {t.name: t for t in tools}
    results = []
    for call in calls:
        spec = by_name.get(call["name"])
        if spec is None or spec.handler is None:
            results.append({"ok": False, "error": "no handler"})
            continue
        try:
            results.append(spec.handler(**call.get("arguments", {})))
        except Exception as exc:  # noqa: BLE001 — surface to caller
            results.append({"ok": False, "error": str(exc)})
    return results


def heuristic_tool_match(prompt: str, tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    """Keyword / slot filler for demos when the LM returns empty calls.

    Not a substitute for a trained model — used so the docs site can show the
    ARC 1 product surface before a competitive checkpoint exists.
    """
    import re as _re

    text = (prompt or "").lower()
    by_name = {t.name: t for t in tools}
    calls: List[Dict[str, Any]] = []

    if "get_weather" in by_name and any(
        w in text for w in ("weather", "temperature", "forecast", "how's it", "hows it")
    ):
        city = None
        for marker in (" in ", " for ", " at "):
            if marker in text:
                city = text.split(marker, 1)[1].strip(" ?.!,")
                city = city.split()[0].strip(",.").title() if city else None
                break
        if not city:
            for tok in (prompt or "").replace("?", " ").split():
                if tok[:1].isupper() and tok.lower() not in ("what's", "what", "how", "the"):
                    city = tok.strip(".,!")
                    break
        if city:
            calls.append({"name": "get_weather", "arguments": {"city": city}})

    if "set_lights" in by_name and any(w in text for w in ("light", "dim", "lamp", "brightness")):
        level = 50
        m = _re.search(r"(\d+)\s*%?", text)
        if m:
            level = int(m.group(1))
        room = "living room"
        for candidate in ("living room", "bedroom", "kitchen", "office"):
            if candidate in text:
                room = candidate
                break
        calls.append({"name": "set_lights", "arguments": {"room": room, "level": level}})

    if "convert_currency" in by_name and any(
        w in text for w in ("convert", "currency", "exchange", "usd", "eur", "php", "gbp", "jpy", "ngn")
    ):
        amount = 100.0
        m = _re.search(r"([\d]+(?:\.\d+)?)", text)
        if m:
            amount = float(m.group(1))
        codes = _re.findall(r"\b(usd|eur|gbp|jpy|php|ngn)\b", text)
        codes = [c.upper() for c in codes]
        frm, to = "USD", "PHP"
        if len(codes) >= 2:
            frm, to = codes[0], codes[1]
        elif len(codes) == 1:
            to = codes[0]
            frm = "USD" if to != "USD" else "EUR"
        # Phrasing: "100 USD to PHP"
        m2 = _re.search(
            r"([\d]+(?:\.\d+)?)\s*(usd|eur|gbp|jpy|php|ngn)\s*(?:to|into|in)\s*(usd|eur|gbp|jpy|php|ngn)",
            text,
        )
        if m2:
            amount = float(m2.group(1))
            frm, to = m2.group(2).upper(), m2.group(3).upper()
        calls.append(
            {
                "name": "convert_currency",
                "arguments": {
                    "amount": amount,
                    "from_currency": frm,
                    "to_currency": to,
                },
            }
        )

    if "send_message" in by_name and any(w in text for w in ("message", "text ", "tell ", "ping ")):
        to = "Alex"
        for name in ("alex", "sam", "jordan", "maya"):
            if name in text:
                to = name.title()
                break
        msg = prompt.strip()
        for prefix in ("message ", "text ", "tell ", "ping "):
            if text.startswith(prefix) or f" {prefix}" in f" {text}":
                # Keep original casing for body after contact name when possible.
                break
        m = _re.search(r"(?:that|saying|:)\s*[\"']?(.+?)[\"']?\s*$", prompt, _re.I)
        if m:
            msg = m.group(1).strip()
        elif " to " in text:
            # "message Maya that dinner is ready"
            after = prompt.split(" to ", 1)[-1]
            parts = after.split(" that ", 1)
            if len(parts) == 2:
                to = parts[0].strip().split()[0].title()
                msg = parts[1].strip()
        calls.append({"name": "send_message", "arguments": {"to": to, "message": msg}})

    return validate_calls_against_tools(calls, tools)


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100, "a": 1, "an": 1,
}
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def coerce_value(ptype: str, text: str) -> Tuple[bool, Any]:
    """Convert a copied span to the parameter's JSON type. ``(ok, value)``."""
    ptype = (ptype or "string").lower()
    if ptype in ("integer", "int", "number", "float"):
        m = _NUMBER_RE.search(text)
        if m:
            try:
                num = float(m.group(0).replace(",", ""))
            except ValueError:
                return False, None
        elif text.strip().lower() in _WORD_NUMBERS:
            num = float(_WORD_NUMBERS[text.strip().lower()])
        else:
            return False, None
        return True, int(round(num)) if ptype in ("integer", "int") else num
    return bool(text), text


def _sigmoid(x, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64) / t))


def _softmax(x, t: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / t
    z = np.exp(z - z.max())
    return z / z.sum()


class SchemaMemory:
    """Cache of schema engrams keyed by schema text.

    Tool, parameter, and option texts rarely change between requests, so ARC 1
    keeps the engram of each text it has perceived. Texts not yet cached are
    perceived inside the same forward pass as the utterance (see
    ``Arc1Model.decide_joint``) and stored here afterwards. Call ``clear()``
    after changing weights.
    """

    def __init__(self, capacity: int = 8192):
        self.capacity = int(capacity)
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, text: str) -> bool:
        return text in self._store

    def clear(self) -> None:
        self._store.clear()

    def get(self, text: str) -> np.ndarray:
        self._store.move_to_end(text)
        return self._store[text]

    def put(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        for text, vec in zip(texts, vectors):
            self._store[text] = vec
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)


@dataclass
class _Probe:
    role: int
    text_a: str
    text_b: Optional[str]
    tool: str
    param: Optional[ToolParam] = None
    option: Optional[str] = None


class Arc1Agent:
    """Tool calling, extraction, and embeddings with ARC 1 Resonant Schema Binding.

    A request is **one** forward pass: the utterance (plus any schema text not
    yet cached) is perceived once, every offered tool, parameter, and enum
    option binds to it in parallel, and the readouts are assembled into
    schema-valid calls or a classification label. Strings and numbers are
    anchored in the user's words; enums are selected; booleans and optional
    arguments fire or stay silent. Probabilities are temperature-calibrated.
    """

    def __init__(
        self,
        model,
        tokenizer: BytePairTokenizer,
        tools: Optional[Sequence[ToolSpec]] = None,
        system: str = "",
        allow_heuristic: bool = False,
        tool_threshold: float = 0.5,
        presence_threshold: float = 0.5,
    ):
        import tensorflow as tf

        from .arc1_codec import Arc1Codec

        self.model = model
        self.tokenizer = tokenizer
        self.tools = list(tools or [])
        self.system = system
        self.allow_heuristic = allow_heuristic
        self.tool_threshold = tool_threshold
        self.presence_threshold = presence_threshold
        cfg = model.arc1_config
        self.codec = Arc1Codec(tokenizer, cfg.seq_len, cfg.schema_len)
        self.memory = SchemaMemory()
        self._joint_fns: Dict[int, Callable] = {}
        d = cfg.d_model

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)], reduce_retracing=True)
        def embed(ids):
            return model.embed_text(ids, training=False)

        self._embed_fn = embed
        self._signature = [
            tf.TensorSpec([None, None], tf.int32),   # row 0 utterance, rows 1.. uncached schema texts
            tf.TensorSpec([None, d], tf.float32),    # cached engrams used by this request
            tf.TensorSpec([None], tf.int32),         # probe_a index into [cached; new]
            tf.TensorSpec([None], tf.int32),         # probe_b index, -1 = no context
            tf.TensorSpec([None], tf.int32),         # roles
        ]

    # ------------------------------------------------------------ primitives
    def _temp(self, readout: str) -> float:
        return self.model.arc1_config.temperature(readout)

    def _joint_fn(self, cycles: int) -> Callable:
        fn = self._joint_fns.get(cycles)
        if fn is None:
            import tensorflow as tf

            model = self.model

            @tf.function(input_signature=self._signature, reduce_retracing=True)
            def fn(ids, bank, probe_a, probe_b, roles):
                return model.decide_joint(ids, bank, probe_a, probe_b, roles, cycles=cycles, training=False)

            self._joint_fns[cycles] = fn
        return fn

    @staticmethod
    def _plan(tools: Sequence[ToolSpec], with_tool_probes: bool = True) -> List[_Probe]:
        from .arc1_codec import ROLE_OPTION, ROLE_TOOL, option_text, param_text, role_for, tool_text

        probes: List[_Probe] = []
        for t in tools:
            ttext = tool_text(t.name, t.description)
            if with_tool_probes:
                probes.append(_Probe(ROLE_TOOL, ttext, None, t.name))
            for p in t.parameters:
                ptext = param_text(p.name, p.type, p.description, p.required)
                probes.append(_Probe(role_for(p.type, p.enum), ptext, ttext, t.name, p))
                hints = p.enum_descriptions or {}
                for opt in p.enum or []:
                    probes.append(_Probe(ROLE_OPTION, option_text(opt, hints.get(str(opt))), ptext, t.name, p, str(opt)))
        return probes

    def _bind(self, text: str, probes: List[_Probe], cycles: Optional[int]):
        """One forward pass: perceive ``text`` and any uncached schema texts together,
        then bind every probe. Returns (utterance, outputs, stats)."""
        import tensorflow as tf

        from .arc1_codec import pad_batch

        cfg = self.model.arc1_config
        cycles = cfg.resolve_cycles(cycles)
        utt = self.codec.utterance(text)
        needed = list(dict.fromkeys([p.text_a for p in probes] + [p.text_b for p in probes if p.text_b]))
        cached = [t for t in needed if t in self.memory]
        missing = [t for t in needed if t not in self.memory]
        index = {t: i for i, t in enumerate(cached + missing)}
        d = cfg.d_model
        bank = np.stack([self.memory.get(t) for t in cached]) if cached else np.zeros((0, d), np.float32)
        ids = pad_batch([utt.ids] + [self.codec.schema(t) for t in missing], max_len=cfg.seq_len)
        probe_a = np.asarray([index[p.text_a] for p in probes], dtype=np.int32)
        probe_b = np.asarray([index[p.text_b] if p.text_b else -1 for p in probes], dtype=np.int32)
        roles = np.asarray([p.role for p in probes], dtype=np.int32)
        out = self._joint_fn(cycles)(
            tf.constant(ids), tf.constant(bank, tf.float32), tf.constant(probe_a), tf.constant(probe_b),
            tf.constant(roles),
        )
        out = {k: v.numpy() for k, v in out.items()}
        self.memory.put(missing, out.pop("new_engrams"))
        self.memory.hits += len(cached)
        self.memory.misses += len(missing)
        stats = {"tokens": len(utt.ids), "probes": len(probes), "cycles": cycles, "forward_passes": 1,
                 "schema_encoded": len(missing), "schema_cached": len(cached)}
        return utt, out, stats

    def _anchor(self, utt, i: int, out: Dict[str, np.ndarray], ptype: str,
                blocked: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Best grounded span for probe ``i``, skipping ``blocked`` token positions."""
        from .arc1_codec import best_span

        start, end = out["anchor_start"][i], out["anchor_end"][i]
        if blocked is not None:
            start = np.where(blocked, -1e9, start)
            end = np.where(blocked, -1e9, end)
            if not (~blocked[utt.lo:utt.hi]).any():
                return {"ok": False, "value": None, "p": 0.0}
        s, e, p_span = best_span(start, end, utt.lo, utt.hi, temperature=self._temp("anchor"))
        surface = self.codec.tokens_to_text(utt, s, e)
        ok, value = coerce_value(ptype, surface)
        return {"ok": ok, "value": value, "surface": surface, "tokens": (s, e),
                "span": self.codec.tokens_to_char_span(utt, s, e), "p": p_span if ok else 0.0}

    def _read_arguments(self, utt, probes: List[_Probe], out: Dict[str, np.ndarray],
                        p_fire: np.ndarray) -> List[Dict[str, Any]]:
        """Decode every parameter probe into a typed, grounded decision."""
        from .arc1_codec import ROLE_BOOL, ROLE_ENUM, ROLE_OPTION, ROLE_SPAN

        options: Dict[Tuple[str, str], List[int]] = {}
        for i, p in enumerate(probes):
            if p.role == ROLE_OPTION:
                options.setdefault((p.tool, p.param.name), []).append(i)
        d = out["select_q"].shape[-1]
        decisions, anchored = [], []
        for i, pr in enumerate(probes):
            if pr.role not in (ROLE_SPAN, ROLE_BOOL, ROLE_ENUM):
                continue
            param = pr.param
            ptype = (param.type or "string").lower()
            p_present = 1.0 if param.required else float(p_fire[i])
            decision: Dict[str, Any] = {"tool": pr.tool, "param": param.name, "p_present": p_present}
            if pr.role == ROLE_BOOL:
                p_true = float(p_fire[i])
                decision.update(kind="fire", present=True, value=p_true >= 0.5,
                                p=max(p_true, 1.0 - p_true), p_true=p_true, p_present=1.0)
            elif pr.role == ROLE_ENUM:
                idx = options.get((pr.tool, param.name), [])
                logits = out["select_k"][idx] @ out["select_q"][i] / np.sqrt(d)
                probs = _softmax(logits, self._temp("select"))
                best = int(np.argmax(probs))
                decision.update(
                    kind="select",
                    present=p_present >= self.presence_threshold,
                    value=str(param.enum[best]),
                    p=float(probs[best]),
                    distribution={str(o): float(p) for o, p in zip(param.enum, probs)},
                )
            elif utt.hi <= utt.lo:
                decision.update(kind="anchor", present=False, value=None, p=0.0)
            else:
                a = self._anchor(utt, i, out, ptype)
                decision.update(kind="anchor", present=a["ok"] and p_present >= self.presence_threshold,
                                value=a["value"], surface=a.get("surface"), span=a.get("span"), p=a["p"])
                anchored.append((decision, i, ptype, param.required, a.get("tokens")))
            decisions.append(decision)
        self._exclusive_anchoring(utt, out, anchored)
        return decisions

    def _exclusive_anchoring(self, utt, out: Dict[str, np.ndarray], anchored) -> None:
        """Within one call a token belongs to at most one argument.

        The most confident anchors claim their tokens first. A later optional
        argument whose span collides is dropped; a required one is re-anchored
        on the tokens that are still free.
        """
        width = out["anchor_start"].shape[-1]
        claimed: Dict[str, np.ndarray] = {}
        live = [x for x in anchored if x[0]["present"] and x[4] is not None]
        for decision, i, ptype, required, (s, e) in sorted(live, key=lambda x: -x[0]["p"] * x[0]["p_present"]):
            taken = claimed.setdefault(decision["tool"], np.zeros(width, dtype=bool))
            if taken[s:e + 1].any():
                if not required:
                    decision.update(present=False, note="span already anchored by a stronger argument")
                    continue
                a = self._anchor(utt, i, out, ptype, blocked=taken)
                decision.update(present=a["ok"], value=a["value"], surface=a.get("surface"),
                                span=a.get("span"), p=a["p"], note="re-anchored on unclaimed tokens")
                if not a["ok"]:
                    continue
                s, e = a["tokens"]
            taken[s:e + 1] = True

    # ------------------------------------------------------------- products
    def run(
        self,
        prompt: str,
        tools: Optional[Sequence[ToolSpec]] = None,
        execute: bool = True,
        cycles: Optional[int] = None,
    ) -> Dict[str, Any]:
        from .arc1_codec import ROLE_TOOL

        t0 = time.perf_counter()
        active = list(tools if tools is not None else self.tools)
        probes = self._plan(active)
        tool_probs: Dict[str, float] = {}
        decisions: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {"tokens": 0, "probes": 0, "cycles": self.model.arc1_config.resolve_cycles(cycles)}
        if probes:
            utt, out, stats = self._bind(prompt, probes, cycles)
            p_fire = _sigmoid(out["fire"], self._temp("fire"))
            tool_probs = {p.tool: float(p_fire[i]) for i, p in enumerate(probes) if p.role == ROLE_TOOL}
            fired = {name for name, p in tool_probs.items() if p >= self.tool_threshold}
            keep = [p.tool in fired for p in probes]
            decisions = self._read_arguments(
                utt, [p for p, k in zip(probes, keep) if k],
                {k: v[np.asarray(keep)] for k, v in out.items()},
                p_fire[np.asarray(keep)],
            ) if fired else []
        selected = sorted(
            (t for t in active if tool_probs.get(t.name, 0.0) >= self.tool_threshold),
            key=lambda t: -tool_probs[t.name],
        )

        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for d in decisions:
            by_tool.setdefault(d["tool"], []).append(d)
        calls, call_conf, notes = [], [], []
        for t in selected:
            params = {p.name: p for p in t.parameters}
            args, conf, missing = {}, tool_probs[t.name], []
            parts = [f"{t.name} fires (p={tool_probs[t.name]:.2f})"]
            for d in by_tool.get(t.name, []):
                required = params[d["param"]].required
                if d["present"]:
                    args[d["param"]] = d["value"]
                    conf *= d["p"] * (1.0 if required else d["p_present"])
                    parts.append(f"{d['param']}={d['value']!r} (p={d['p']:.2f})")
                elif required:
                    missing.append(d["param"])
                else:
                    conf *= 1.0 - d["p_present"]
            if missing:
                notes.append(f"{t.name} held back: could not anchor required {', '.join(missing)}")
                continue
            calls.append({"name": t.name, "arguments": args})
            call_conf.append(conf)
            notes.append("; ".join(parts))

        calls = validate_calls_against_tools(calls, active)
        source = "model"
        if not calls and self.allow_heuristic:
            heur = heuristic_tool_match(prompt, active)
            if heur:
                calls, source = heur, "heuristic"
                notes.append("keyword fallback (allow_heuristic=True)")

        if calls and source == "model":
            confidence: Optional[float] = float(min(call_conf))
        elif calls:
            confidence = None  # heuristic calls carry no calibrated probability
        else:
            confidence = float(1.0 - max(tool_probs.values(), default=0.0))
            notes.append("no offered tool fires")

        results = execute_tools(calls, active) if execute else []
        payload = {"tools": tool_probs, "arguments": decisions}
        return {
            "reasoning": ". ".join(notes) + ".",
            "function_calls": calls,
            "results": results,
            "confidence": confidence,
            "raw": json.dumps(payload, default=str),
            "decisions": payload,
            "source": source,
            "cycles": stats["cycles"],
            "stats": stats,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def extract(self, text: str, record_schema: Dict[str, Any], cycles: Optional[int] = None) -> Dict[str, Any]:
        from .arc1_data import EXTRACT_TOOL_DESCRIPTION, EXTRACT_TOOL_NAME

        t0 = time.perf_counter()
        params = []
        for key, spec in record_schema.items():
            spec = spec if isinstance(spec, dict) else {"type": "string", "description": str(spec or key)}
            params.append(
                ToolParam(
                    name=key,
                    type=str(spec.get("type", "string")),
                    description=str(spec.get("description", key)),
                    required=False,
                    enum=spec.get("enum"),
                )
            )
        spec = ToolSpec(EXTRACT_TOOL_NAME, EXTRACT_TOOL_DESCRIPTION, params)
        probes = self._plan([spec], with_tool_probes=False)
        decisions, stats = [], {"tokens": 0, "probes": 0, "cycles": self.model.arc1_config.resolve_cycles(cycles)}
        if probes:
            utt, out, stats = self._bind(text, probes, cycles)
            decisions = self._read_arguments(utt, probes, out, _sigmoid(out["fire"], self._temp("fire")))
        record, confs, notes = {}, [], []
        for d in decisions:
            if d["present"]:
                record[d["param"]] = d["value"]
                confs.append(d["p"] * d["p_present"])
                notes.append(f"{d['param']}={d['value']!r} (p={d['p']:.2f})")
            else:
                confs.append(1.0 - d["p_present"])
                notes.append(f"{d['param']} not found (p_absent={1.0 - d['p_present']:.2f})")
        calls = [{"name": EXTRACT_TOOL_NAME, "arguments": record}] if record else []
        return {
            "record": record,
            "function_calls": calls,
            "confidence": float(min(confs)) if confs else 0.0,
            "reasoning": "; ".join(notes) + ".",
            "raw": json.dumps(decisions, default=str),
            "decisions": decisions,
            "source": "model",
            "cycles": stats["cycles"],
            "stats": stats,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def classify(self, text: str, labels: Sequence[str], task: Optional[str] = None,
                 cycles: Optional[int] = None, descriptions: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Pick one of ``labels`` for ``text`` in one forward pass.

        The label set is a classification schema: one argument whose options are
        the labels, so each label is a probe that resonates with the text and
        the ``select`` readout returns a calibrated distribution over them.
        ``task`` optionally says what the labels mean ("Route the ticket to a team"), and
        ``descriptions`` optionally gives each label a hint ({"billing": "charges, refunds"}).
        """
        from .arc1_data import classify_tool_spec

        t0 = time.perf_counter()
        labels = [str(x) for x in dict.fromkeys(labels) if str(x).strip()]
        if not labels:
            raise ValueError("classify() needs at least one label")
        spec = classify_tool_spec(labels, task, descriptions)
        utt, out, stats = self._bind(text, self._plan([spec], with_tool_probes=False), cycles)
        decision = self._read_arguments(utt, self._plan([spec], with_tool_probes=False), out,
                                        _sigmoid(out["fire"], self._temp("fire")))[0]
        dist = decision["distribution"]
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        return {
            "label": decision["value"],
            "confidence": float(decision["p"]),
            "distribution": dict(ranked),
            "reasoning": ", ".join(f"{k} (p={v:.2f})" for k, v in ranked[:3]) + ".",
            "source": "model",
            "cycles": stats["cycles"],
            "stats": stats,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def embed(self, text: str) -> List[float]:
        """Attention-pooled utterance field, L2-normalised."""
        import tensorflow as tf

        from .arc1_codec import pad_batch

        ids = pad_batch([self.codec.utterance(text).ids], max_len=self.model.arc1_config.seq_len)
        vec = self._embed_fn(tf.constant(ids))
        return [float(x) for x in vec.numpy()[0].tolist()]
