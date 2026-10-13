"""ARCANE 5.0: a complete grounded System 1 decision loop."""

from gpbacay_arcane import DecisionSpace, System1


def support_router(text, labels):
    """A tiny stand-in for ARC 1 or a trained direct classifier."""
    text = text.lower()
    scores = {label: 0.05 for label in labels}
    if any(word in text for word in ("charged", "invoice", "refund")):
        scores["billing"] = 0.8
    elif any(word in text for word in ("parcel", "delivery", "tracking")):
        scores["shipping"] = 0.8
    else:
        scores = {label: 0.2 for label in labels}
    return scores


model = System1(
    support_router,
    DecisionSpace(
        ["billing", "shipping", "other"],
        min_confidence=0.60,
        min_margin=0.15,
    ),
    memory_weight=0.65,
)

# Teach a correction in one example; no optimizer or fine-tuning is involved.
model.remember("the tracking page says delivered but nothing arrived", "shipping")

for request in (
    "why was my card charged twice?",
    "tracking says delivered but the parcel is missing",
    "hello, I need some help",
):
    decision = model.decide(request)
    print(
        {
            "input": request,
            "label": decision.label,
            "confidence": round(decision.confidence, 3),
            "abstained": decision.abstained,
            "source": decision.source,
            "evidence": [item["text"] for item in decision.evidence],
        }
    )
