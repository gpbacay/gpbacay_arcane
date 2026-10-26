"""Hosted ARCANE SLM: open-model replies, System 1 decisions, no training."""

from gpbacay_arcane.hosted_slm import ArcaneHostedSLM, scores_from_reply


class Scripted:
    def __init__(self, reply):
        self.reply = reply
        self.base_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.prompts = []

    def complete(self, prompt, max_new_tokens=64, temperature=0.0, messages=None):
        self.prompts.append(prompt)
        self.messages = messages
        return self.reply


def test_scores_from_reply_picks_one_label_and_abstains_otherwise():
    labels = ["billing", "shipping", "other"]
    assert scores_from_reply("billing", labels) == {"billing": 1.0, "shipping": 0.0, "other": 0.0}
    assert scores_from_reply("I am not sure", labels) == {"billing": 0.0, "shipping": 0.0, "other": 0.0}
    assert scores_from_reply("billing or shipping", labels)["billing"] == 0.5
    assert scores_from_reply("the shipment left", labels)["shipping"] == 0.0


def test_decide_uses_the_open_model_reply_and_records_its_license():
    model = ArcaneHostedSLM(["billing", "shipping", "other"], Scripted("billing"))
    decision = model.decide("why was my card charged twice?")

    assert decision.label == "billing"
    assert decision.metadata["base_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert decision.metadata["license"] == "Apache-2.0"
    assert "billing" in model.completer.prompts[0]


def test_unknown_reply_abstains():
    model = ArcaneHostedSLM(["billing", "shipping"], Scripted("hello there"))
    decision = model.decide("something new")
    assert decision.abstained and decision.reason == "no_supported_decision"


def test_remember_overrides_a_similar_cue_without_training():
    model = ArcaneHostedSLM(
        ["billing", "shipping"],
        Scripted("shipping"),
        memory_weight=0.8,
        min_confidence=0.55,
        min_margin=0.0,
    )
    model.remember("my card was charged twice", "billing")
    decision = model.decide("why did you charge my card twice")
    assert decision.label == "billing"
    assert decision.source == "model+memory"


def test_answer_returns_the_completer_text():
    model = ArcaneHostedSLM(["other"], Scripted("gravity pulls masses together"))
    assert model.answer("What is gravity?") == "gravity pulls masses together"


def test_answer_sends_prior_turns_before_the_new_prompt():
    scripted = Scripted("ok")
    model = ArcaneHostedSLM(["other"], scripted)
    model.answer("second", history=[("user", "first"), ("assistant", "reply"), ("system", "ignore")])
    assert scripted.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
