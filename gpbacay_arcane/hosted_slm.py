"""System 1 SLM whose language weights stay an open model.

SmolLM2-135M-Instruct (Apache-2.0) runs through a local Ollama server.
ARCANE supplies the closed label set, abstention, and example memory.
Nothing here trains or copies those weights.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, Optional, Sequence

from .system1 import DecisionSpace, GroundedDecision, System1

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_MODEL_LICENSE = "Apache-2.0"
DEFAULT_OLLAMA_MODEL = "smollm2:135m"


def scores_from_reply(reply: str, labels: Sequence[str]) -> Dict[str, float]:
    """Map a short reply onto the closed label set.

    ponytail: whole-word match, not next-token logprobs. Switch when replies hedge.
    """
    folded = reply.casefold()
    hits = [
        label
        for label in labels
        if re.search(rf"(?<!\w){re.escape(label.casefold())}(?!\w)", folded)
    ]
    if len(hits) == 1:
        chosen = hits[0]
        return {label: 1.0 if label == chosen else 0.0 for label in labels}
    if len(hits) > 1:
        share = 1.0 / len(hits)
        return {label: share if label in hits else 0.0 for label in labels}
    return {label: 0.0 for label in labels}


def _label_prompt(text: str, labels: Sequence[str]) -> str:
    listed = ", ".join(labels)
    return (
        "Choose exactly one label from this list: "
        f"{listed}.\n"
        "Reply with that label and no other words.\n\n"
        f"Text:\n{text}"
    )


class OllamaCompleter:
    """Talk to a local Ollama server. Stdlib HTTP, no extra package."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = "http://127.0.0.1:11434",
        base_model_id: str = BASE_MODEL_ID,
        timeout: float = 180.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.base_model_id = base_model_id
        self.timeout = timeout

    def complete(self, prompt: str, *, max_new_tokens: int = 64, temperature: float = 0.0, messages=None) -> str:
        turns = list(messages) if messages is not None else [{"role": "user", "content": prompt}]
        payload = json.dumps({
            "model": self.model,
            "messages": turns,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": int(max_new_tokens)},
        }).encode()
        request = urllib.request.Request(
            self.host + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.host}. "
                f"Start Ollama and run: ollama pull {self.model}"
            ) from exc
        return str(body["message"]["content"])


class ArcaneHostedSLM:
    """ARCANE decisions on top of an already-trained open language model."""

    def __init__(
        self,
        labels: Sequence[str],
        completer=None,
        *,
        min_confidence: float = 0.6,
        min_margin: float = 0.15,
        memory_weight: float = 0.65,
    ):
        self.completer = completer if completer is not None else OllamaCompleter()
        self.base_model_id = getattr(self.completer, "base_model_id", BASE_MODEL_ID)
        self.system = System1(
            self,
            DecisionSpace(labels, min_confidence=min_confidence, min_margin=min_margin),
            memory_weight=memory_weight,
        )

    def classify(self, text: str, labels: Sequence[str], **kwargs) -> Dict[str, object]:
        del kwargs
        reply = self.completer.complete(_label_prompt(text, labels), max_new_tokens=16, temperature=0.0)
        return {
            "distribution": scores_from_reply(reply, labels),
            "base_model": self.base_model_id,
            "license": BASE_MODEL_LICENSE,
            "reply": reply,
        }

    def answer(self, prompt: str, *, max_new_tokens: int = 128, temperature: float = 0.2, history=None) -> str:
        """Free-form reply from the open model. This path does not use memory."""
        turns = [
            {"role": role, "content": content}
            for role, content in (history or [])
            if role in ("user", "assistant") and str(content).strip()
        ]
        turns.append({"role": "user", "content": prompt})
        return self.completer.complete(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, messages=turns,
        )

    def decide(self, text: str) -> GroundedDecision:
        return self.system.decide(text)

    def remember(self, text: str, label: Optional[str]) -> str:
        return self.system.remember(text, label)
