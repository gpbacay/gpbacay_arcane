#!/usr/bin/env python3
"""ARCANE SLM with no local training.

Language weights: HuggingFaceTB/SmolLM2-135M-Instruct (Apache-2.0), via Ollama.
One-time setup:  ollama pull smollm2:135m

  python examples/chat_hosted_slm.py --demo
  python examples/chat_hosted_slm.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.hosted_slm import ArcaneHostedSLM

LABELS = ["billing", "shipping", "other"]


class _Demo:
    base_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def complete(self, prompt, max_new_tokens=64, temperature=0.0):
        text = prompt.lower()
        if "choose exactly one label" in text:
            if "charg" in text or "invoice" in text or "refund" in text:
                return "billing"
            if "parcel" in text or "deliver" in text or "tracking" in text:
                return "shipping"
            return "unsure"
        return "SmolLM2 would answer here. This demo completer only labels."


def main():
    parser = argparse.ArgumentParser(description="ARCANE hosted SLM")
    parser.add_argument("--demo", action="store_true", help="Scripted replies, no Ollama")
    args = parser.parse_args()
    model = ArcaneHostedSLM(LABELS, _Demo() if args.demo else None)
    print(f"base model: {model.base_model_id} ({'demo' if args.demo else 'ollama'})")
    print("Commands: a question, 'remember <text> => <label>', or blank to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line.lower().startswith("remember ") and "=>" in line:
            text, label = line.split("=>", 1)
            text = text[len("remember "):].strip()
            label = label.strip()
            model.remember(text, label)
            print(f"remembered {label!r}")
            continue
        decision = model.decide(line)
        print(decision.to_dict())


if __name__ == "__main__":
    main()
