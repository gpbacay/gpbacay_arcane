"""Tool calling and structured extraction for ARC 1."""

from __future__ import annotations

import json
import re
import time
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
        for needle in (" in ", " for ", " at "):
            if needle in text:
                city = text.split(needle, 1)[1].strip(" ?.!,")
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


def json_token_allowlist(tokenizer: BytePairTokenizer, vocab_size: int) -> List[int]:
    """Prefer printable / JSON-ish tokens for constrained decode."""
    if hasattr(tokenizer, "generation_ids"):
        return list(tokenizer.generation_ids(printable_only=True))
    return list(range(2, min(vocab_size, 512)))


def _sigmoid(x, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64) / t))


def _softmax(x, t: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / t
    z = np.exp(z - z.max())
    return z / z.sum()


class Arc1Agent:
    """Tool calling, extraction, and embeddings over ``Arc1Model`` decision heads.

    Each request is two batched forward passes: (1) one ``noul`` sequence per
    offered tool decides which tools apply; (2) one sequence per argument of the
    selected tools fills values — copied spans for strings/numbers, ``choice``
    over enum options, ``noul`` for booleans and optional-argument presence.
    Output is always schema-valid; probabilities are temperature-calibrated.
    """

    def __init__(
        self,
        model,
        tokenizer: BytePairTokenizer,
        tools: Optional[Sequence[ToolSpec]] = None,
        system: str = "",
        temperature: float = 0.2,
        max_new_tokens: int = 96,
        allow_heuristic: bool = False,
        tool_threshold: float = 0.5,
        presence_threshold: float = 0.5,
    ):
        from .arc1_codec import Arc1Codec

        self.model = model
        self.tokenizer = tokenizer
        self.tools = list(tools or [])
        self.system = system
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.allow_heuristic = allow_heuristic
        self.tool_threshold = tool_threshold
        self.presence_threshold = presence_threshold
        self.codec = Arc1Codec(tokenizer, model.arc1_config.seq_len)

    # ------------------------------------------------------------ primitives
    def _forward(self, seqs, depth: Optional[int]) -> Dict[str, np.ndarray]:
        import tensorflow as tf

        from .arc1_codec import pad_batch

        batch = pad_batch([s.ids for s in seqs], max_len=self.model.arc1_config.seq_len)
        out = self.model.decide(tf.constant(batch), training=False, depth=depth)
        return {k: v.numpy() for k, v in out.items() if k != "hidden"}

    def _temp(self, head: str) -> float:
        return self.model.arc1_config.temperature(head)

    def score_tools(self, prompt: str, tools: Sequence[ToolSpec], depth: Optional[int] = None) -> Dict[str, float]:
        """Laya ``noul`` per tool: calibrated P(tool applies to the prompt)."""
        if not tools:
            return {}
        seqs = [self.codec.tool_seq(t.name, t.description, prompt) for t in tools]
        probs = _sigmoid(self._forward(seqs, depth)["noul"], self._temp("noul"))
        return {t.name: float(p) for t, p in zip(tools, probs)}

    def fill_arguments(
        self,
        text: str,
        requests: Sequence[Tuple[str, ToolParam]],
        depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Decide every (tool_name, param) in one batch.

        Returns one dict per request with ``present``, ``value``, ``p`` (calibrated
        probability of the value), ``p_present``, and ``kind``.
        """
        from .arc1_codec import best_span

        seqs, enum_by_idx = [], {}
        for tool_name, param in requests:
            seqs.append(
                self.codec.arg_seq(tool_name, param.name, param.type, param.description, param.required, text)
            )
        for idx, (tool_name, param) in enumerate(requests):
            if param.enum:
                start = len(seqs)
                for option in param.enum:
                    seqs.append(self.codec.enum_seq(tool_name, param.name, param.description, str(option), text))
                enum_by_idx[idx] = (start, len(seqs))
        if not seqs:
            return []
        out = self._forward(seqs, depth)
        p_noul = _sigmoid(out["noul"], self._temp("noul"))

        decisions = []
        for idx, (tool_name, param) in enumerate(requests):
            seq = seqs[idx]
            ptype = (param.type or "string").lower()
            p_present = 1.0 if param.required else float(p_noul[idx])
            decision: Dict[str, Any] = {"tool": tool_name, "param": param.name, "p_present": p_present}
            if ptype in ("boolean", "bool"):
                p_true = float(p_noul[idx])
                decision.update(kind="noul", present=True, value=p_true >= 0.5,
                                p=max(p_true, 1.0 - p_true), p_true=p_true, p_present=1.0)
            elif param.enum:
                a, b = enum_by_idx[idx]
                probs = _softmax(out["choice"][a:b], self._temp("choice"))
                best = int(np.argmax(probs))
                decision.update(
                    kind="choice",
                    present=p_present >= self.presence_threshold,
                    value=str(param.enum[best]),
                    p=float(probs[best]),
                    distribution={str(o): float(p) for o, p in zip(param.enum, probs)},
                )
            elif seq.span_hi <= seq.span_lo:
                decision.update(kind="span", present=False, value=None, p=0.0)
            else:
                s, e, p_span = best_span(
                    out["span_start"][idx], out["span_end"][idx], seq.span_lo, seq.span_hi,
                    temperature=self._temp("span"),
                )
                surface = self.codec.tokens_to_text(seq, s, e)
                ok, value = coerce_value(ptype, surface)
                decision.update(
                    kind="span",
                    present=ok and p_present >= self.presence_threshold,
                    value=value,
                    surface=surface,
                    p=p_span if ok else 0.0,
                )
            decisions.append(decision)
        return decisions

    # ------------------------------------------------------------- products
    def run(
        self,
        prompt: str,
        tools: Optional[Sequence[ToolSpec]] = None,
        execute: bool = True,
        depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        active = list(tools if tools is not None else self.tools)
        tool_probs = self.score_tools(prompt, active, depth)
        selected = sorted(
            (t for t in active if tool_probs.get(t.name, 0.0) >= self.tool_threshold),
            key=lambda t: -tool_probs[t.name],
        )
        requests = [(t.name, p) for t in selected for p in t.parameters]
        decisions = self.fill_arguments(prompt, requests, depth) if requests else []

        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for d in decisions:
            by_tool.setdefault(d["tool"], []).append(d)
        calls, call_conf, notes = [], [], []
        for t in selected:
            params = {p.name: p for p in t.parameters}
            args, conf, missing = {}, tool_probs[t.name], []
            parts = [f"{t.name} applies (p={tool_probs[t.name]:.2f})"]
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
                notes.append(f"{t.name} skipped: could not ground required {', '.join(missing)}")
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
            notes.append("no offered tool applies")

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
            "depth": self.model.arc1_config.resolve_depth(depth),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

    def extract(self, text: str, record_schema: Dict[str, Any], depth: Optional[int] = None) -> Dict[str, Any]:
        from .arc1_data import EXTRACT_TOOL_NAME

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
        decisions = self.fill_arguments(text, [(EXTRACT_TOOL_NAME, p) for p in params], depth)
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
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

    def embed(self, text: str, depth: Optional[int] = None) -> List[float]:
        """Echo embedding: mean-pool over the COPY half of ``TEXT: x / COPY: x``."""
        import tensorflow as tf

        from .arc1_codec import pad_batch

        seq = self.codec.embed_seq(text)
        ids = pad_batch([seq.ids])
        mask = np.zeros_like(ids, dtype=bool)
        mask[0, seq.span_lo : seq.span_hi] = True
        vec = self.model.embed_text(tf.constant(ids), training=False, depth=depth, pool_mask=tf.constant(mask))
        return [float(x) for x in vec.numpy()[0].tolist()]


