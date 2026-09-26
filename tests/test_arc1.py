"""Tests for ARC 1 (Resonant Schema Binding): mechanisms, model, codec, data, training, agent."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel, Arc1Model
from gpbacay_arcane.arc1_codec import (
    ROLE_BOOL,
    ROLE_ENUM,
    ROLE_SPAN,
    ROLE_TOOL,
    Arc1Codec,
    best_span,
    pad_batch,
    role_for,
)
from gpbacay_arcane.arc1_data import build_tool_library, sample_extract_example, sample_tool_example
from gpbacay_arcane.arc1_train import BATCH_SIGNATURE, compute_losses, sample_batch
from gpbacay_arcane.layers import Arc1PerceptionBlock, ResonantChannelMixer
from gpbacay_arcane.mechanisms import ConceptEngram, FieldAttention, FieldResonance, ResonantBinding
from gpbacay_arcane.tokenization import BytePairTokenizer
from gpbacay_arcane.tools import (
    Arc1Agent,
    ToolParam,
    ToolSpec,
    coerce_value,
    heuristic_tool_match,
    parse_agent_json,
    validate_calls_against_tools,
)

SMALL = dict(vocab_size=512, d_model=32, num_layers=2, num_heads=2, seq_len=128, schema_len=48,
             engram_table_size=256, engram_rows=2, dropout_rate=0.0, binding_heads=2, binding_cycles=2)


def _small_model():
    return Arc1Model(Arc1Config(**SMALL)).build_model()


def _probe_inputs(model, n=3):
    d = model.arc1_config.d_model
    rng = np.random.RandomState(0)
    a = tf.constant(rng.randn(n, d).astype(np.float32))
    b = tf.constant(rng.randn(n, d).astype(np.float32))
    roles = tf.constant([ROLE_TOOL, ROLE_SPAN, ROLE_BOOL][:n], dtype=tf.int32)
    return a, b, roles


# ------------------------------------------------------------------ mechanisms
def test_field_attention_is_bidirectional_and_pad_invariant():
    layer = FieldAttention(d_model=16, num_heads=2)
    x = tf.random.normal((1, 6, 16), seed=1)
    mask = tf.constant([[True] * 6])
    y = layer(x, token_mask=mask)
    # Changing the last token changes the first token's output (no causal mask).
    x2 = tf.concat([x[:, :-1], x[:, -1:] + 1.0], axis=1)
    assert not np.allclose(y[0, 0].numpy(), layer(x2, token_mask=mask)[0, 0].numpy())
    padded = tf.concat([x, tf.random.normal((1, 4, 16))], axis=1)
    pmask = tf.constant([[True] * 6 + [False] * 4])
    np.testing.assert_allclose(y.numpy(), layer(padded, token_mask=pmask)[:, :6].numpy(), atol=1e-5)


def test_field_resonance_ignores_padding():
    layer = FieldResonance(d_model=8)
    x = tf.random.normal((1, 5, 8), seed=2)
    y = layer(x, token_mask=tf.constant([[True] * 5]))
    padded = tf.concat([x, 100.0 * tf.ones((1, 3, 8))], axis=1)
    y2 = layer(padded, token_mask=tf.constant([[True] * 5 + [False] * 3]))
    np.testing.assert_allclose(y.numpy(), y2[:, :5].numpy(), atol=1e-5)


def test_resonant_binding_shapes_and_cycles_change_state():
    bind = ResonantBinding(d_model=16, num_heads=2, cycles=3)
    field = tf.random.normal((2, 7, 16), seed=3)
    mask = tf.constant([[True] * 7, [True] * 4 + [False] * 3])
    keys, values = bind.field_memory(field)
    probes = tf.random.normal((5, 16), seed=4)
    example = tf.constant([0, 0, 1, 1, 1])
    p3, heard, peak = bind(probes, keys, values, mask, example, cycles=3)
    p1, _, _ = bind(probes, keys, values, mask, example, cycles=1)
    assert p3.shape == (5, 16) and heard.shape == (5, 16) and peak.shape == (5, 2)
    assert not np.allclose(p1.numpy(), p3.numpy())


def test_resonant_binding_never_hears_padding():
    bind = ResonantBinding(d_model=16, num_heads=2, cycles=2)
    field = tf.random.normal((1, 4, 16), seed=5)
    probes = tf.random.normal((2, 16), seed=6)
    example = tf.zeros((2,), tf.int32)
    k, v = bind.field_memory(field)
    short = bind(probes, k, v, tf.constant([[True] * 4]), example)[0]
    padded = tf.concat([field, 50.0 * tf.ones((1, 3, 16))], axis=1)
    k2, v2 = bind.field_memory(padded)
    long = bind(probes, k2, v2, tf.constant([[True] * 4 + [False] * 3]), example)[0]
    np.testing.assert_allclose(short.numpy(), long.numpy(), atol=1e-4)


def test_concept_engram_rows_are_independent_hashes():
    layer = ConceptEngram(d_model=8, table_size=4096, rows_per_token=4, ngram_sizes=(2, 3))
    ids = tf.constant(np.random.RandomState(0).randint(4, 260, size=(1, 64)), dtype=tf.int32)
    keys = layer._lookup_keys(ids).numpy()[0]  # (T, R)
    assert keys.min() >= 0 and keys.max() < 4096
    assert np.mean(np.abs(np.diff(keys, axis=-1)) <= 3) < 0.05


def test_perception_block_and_mixer_shapes():
    assert ResonantChannelMixer(d_model=32, rank_div=4)(tf.random.normal((2, 8, 32))).shape == (2, 8, 32)
    block = Arc1PerceptionBlock(d_model=32, num_heads=4, use_engram=True, engram_table_size=64, engram_rows=2)
    ids = tf.constant([[3, 10, 11, 12, 0, 0, 0, 0]])
    y = block(tf.random.normal((1, 8, 32)), token_ids=ids, token_mask=tf.not_equal(ids, 0))
    assert y.shape == (1, 8, 32)


# ----------------------------------------------------------------------- model
def test_model_readout_shapes_and_embedding_norm():
    model = _small_model()
    utter = tf.constant(pad_batch([[3, 10, 11, 12, 13]]))
    a, b, roles = _probe_inputs(model)
    out = model.decide(utter, tf.zeros((3,), tf.int32), a, b, roles, with_embedding=True)
    assert out["fire"].shape == (3,)
    assert out["anchor_start"].shape == (3, utter.shape[1])
    assert out["select_q"].shape == (3, model.arc1_config.d_model)
    emb = model.embed_text(utter)
    np.testing.assert_allclose(tf.norm(emb, axis=-1).numpy(), 1.0, atol=1e-4)
    eng = model.schema_engrams(tf.constant(pad_batch([[3, 20, 21], [3, 30]])))
    assert eng.shape == (2, model.arc1_config.d_model)


def test_anchors_never_point_at_bos_or_padding():
    model = _small_model()
    utter = tf.constant([[3, 10, 11, 12, 0, 0, 0, 0]])
    a, b, roles = _probe_inputs(model)
    out = model.decide(utter, tf.zeros((3,), tf.int32), a, b, roles)
    start = out["anchor_start"].numpy()
    assert (start[:, 0] < -1e8).all() and (start[:, 4:] < -1e8).all()
    assert (start[:, 1:4] > -1e8).all()


def test_decisions_ignore_right_padding():
    model = _small_model()
    a, b, roles = _probe_inputs(model)
    seq = [3, 40, 41, 42, 43, 44]
    short = model.decide(tf.constant([seq]), tf.zeros((3,), tf.int32), a, b, roles)
    padded = model.decide(tf.constant([seq + [0] * 26]), tf.zeros((3,), tf.int32), a, b, roles)
    np.testing.assert_allclose(short["fire"].numpy(), padded["fire"].numpy(), atol=1e-4)
    np.testing.assert_allclose(short["anchor_start"].numpy(), padded["anchor_start"].numpy()[:, :6], atol=1e-3)
    e1 = model.embed_text(tf.constant([seq])).numpy()
    e2 = model.embed_text(tf.constant([seq + [0] * 26])).numpy()
    np.testing.assert_allclose(e1, e2, atol=1e-4)
    s1 = model.schema_engrams(tf.constant([seq])).numpy()
    s2 = model.schema_engrams(tf.constant([seq + [0] * 10])).numpy()
    np.testing.assert_allclose(s1, s2, atol=1e-4)


def test_cycles_are_per_call_and_stateless():
    model = _small_model()
    utter = tf.constant([[3, 5, 6, 7]])
    a, b, roles = _probe_inputs(model)
    full = model.decide(utter, tf.zeros((3,), tf.int32), a, b, roles)["fire"].numpy()
    fast = model.decide(utter, tf.zeros((3,), tf.int32), a, b, roles, cycles=1)["fire"].numpy()
    assert not np.allclose(full, fast)
    assert model.arc1_config.resolve_cycles() == model.arc1_config.binding_cycles
    assert model.arc1_config.resolve_cycles(99) == model.arc1_config.binding_cycles
    assert model.arc1_config.with_cycles(1).resolve_cycles() == 1


def test_probes_of_different_utterances_are_independent():
    model = _small_model()
    utter = tf.constant(pad_batch([[3, 10, 11, 12], [3, 50, 51, 52, 53, 54]]))
    a, b, roles = _probe_inputs(model, 2)
    both = model.decide(utter, tf.constant([0, 1]), a, b, roles)["fire"].numpy()
    first = model.decide(utter[:1, :4], tf.constant([0]), a[:1], b[:1], roles[:1])["fire"].numpy()
    np.testing.assert_allclose(both[0], first[0], atol=1e-4)


def test_config_roundtrip_keeps_calibration():
    cfg = Arc1Config(**SMALL)
    cfg.calibration = {"fire": 1.7, "select": 2.0, "anchor": 0.9}
    again = Arc1Config.from_dict(cfg.to_dict())
    assert again.calibration == cfg.calibration
    assert again.binding_cycles == 2
    assert Arc1Config.from_dict({"d_model": 32, "unknown_key": 1}).d_model == 32


def test_model_is_lightweight():
    model = Arc1Model.from_preset("arc1-tiny").build_model()
    assert model.count_params() < 2_000_000


# ----------------------------------------------------------------------- codec
def test_codec_span_roundtrip_with_bpe():
    tok = BytePairTokenizer(vocab_size=320).train(["weather in San Francisco and weather in Lagos " * 20])
    codec = Arc1Codec(tok, 256)
    user = "what's the weather in San Francisco today?"
    a = user.index("San Francisco")
    utt = codec.utterance(user)
    s, e = codec.char_span_to_tokens(utt, user, (a, a + len("San Francisco")))
    assert utt.lo <= s <= e < utt.hi
    assert codec.tokens_to_text(utt, s, e) == "San Francisco"
    c0, c1 = codec.tokens_to_char_span(utt, s, e)
    assert user[c0:c1] == "San Francisco"


def test_roles_and_best_span():
    assert role_for("string", None) == ROLE_SPAN
    assert role_for("boolean", None) == ROLE_BOOL
    assert role_for("string", ["a", "b"]) == ROLE_ENUM
    start = np.array([0, 5, 0, 0], dtype=np.float32)
    end = np.array([9, 0, 0, 6], dtype=np.float32)  # end 0 is before start 1
    s, e, p = best_span(start, end, 0, 4)
    assert s <= e and 0 < p <= 1


# ------------------------------------------------------------------------ data
def test_synthetic_spans_match_arguments():
    lib = build_tool_library()
    rng = random.Random(3)
    for _ in range(300):
        ex = sample_tool_example(rng, lib)
        for call in ex.calls:
            for name, (a, b) in call.spans.items():
                surface = ex.user[a:b]
                value = call.arguments[name]
                if isinstance(value, str):
                    assert surface == value
                else:
                    assert float(coerce_value("number", surface)[1]) == float(value)
    for _ in range(100):
        ex = sample_extract_example(rng)
        for k, (a, b) in ex.spans.items():
            if k in ex.record and isinstance(ex.record[k], str):
                assert ex.text[a:b] == ex.record[k]


def test_synthesized_values_keep_shape_and_eval_stays_real():
    from gpbacay_arcane.arc1_data import CITIES, fixed_eval_set, synthesize_like

    rng = random.Random(0)
    for value in ["PR102", "San Francisco", "+63 917 555 0199", "UBER", "alex@example.com"]:
        out = synthesize_like(rng, value)
        assert len(out.split()) == len(value.split())
        assert [c.isdigit() for c in out if not c.isalpha()] == [c.isdigit() for c in value if not c.isalpha()]
    lib = build_tool_library()
    for ex in fixed_eval_set("eval", 200):
        for call in ex.calls:
            if call.tool == "get_weather":
                assert call.arguments["city"].lower() in {c.lower() for c in CITIES["eval"]}
    assert all(t.name in lib for ex in fixed_eval_set("eval", 50) for t in ex.tools)


def test_procedural_tools_have_grounded_spans():
    from gpbacay_arcane.arc1_data import _single, procedural_library

    rng = random.Random(5)
    for _ in range(40):
        for tool in procedural_library(rng, 4).values():
            text, call = _single(rng, tool, "train")
            for name, (a, b) in call.spans.items():
                value = call.arguments[name]
                assert text[a:b] == value if isinstance(value, str) else coerce_value("number", text[a:b])[0]


def test_batch_labels_point_inside_their_utterance():
    tok = BytePairTokenizer(512)
    codec = Arc1Codec(tok, 128, 48)
    batch = sample_batch(random.Random(1), codec, build_tool_library(), n_tool=8, n_extract=4, n_embed=3)
    ids = batch["utter_ids"]
    for idx, s, e in zip(batch["anchor_idx"], batch["anchor_start"], batch["anchor_end"]):
        row = ids[batch["probe_example"][idx]]
        assert 1 <= s <= e < int((row != 0).sum())
    assert batch["probe_a"].max() < len(batch["schema_ids"])
    assert (batch["probe_b"] < len(batch["schema_ids"])).all()


# -------------------------------------------------------------------- training
def test_training_step_reduces_loss_on_fixed_batch():
    model = _small_model()
    codec = Arc1Codec(BytePairTokenizer(512), model.arc1_config.seq_len, model.arc1_config.schema_len)
    batch = sample_batch(random.Random(0), codec, build_tool_library(), n_tool=3, n_extract=1, n_embed=3)
    batch = {k: tf.constant(v) for k, v in batch.items()}
    opt = tf.keras.optimizers.Adam(3e-3)
    opt.build(model.trainable_variables)

    @tf.function(input_signature=[BATCH_SIGNATURE])
    def step(b):
        with tf.GradientTape() as tape:
            losses, _ = compute_losses(model, b, cycles=2, training=True)
        grads = tape.gradient(losses["total"], model.trainable_variables)
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return losses["total"]

    first = float(step(batch))
    for _ in range(25):
        last = float(step(batch))
    assert last < first * 0.7, (first, last)


# ----------------------------------------------------------------------- agent
def test_agent_output_is_schema_valid_even_untrained():
    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))
    tools = [
        ToolSpec("get_weather", "Get the weather.", [ToolParam("city")], handler=lambda city: {"city": city}),
        ToolSpec("set_mode", "Set mode.", [ToolParam("mode", enum=["a", "b"]), ToolParam("on", type="boolean")]),
    ]
    agent.tool_threshold = 0.0  # force every tool through argument reading
    out = agent.run("what's the weather in Lagos?", tools=tools, execute=False, cycles=1)
    assert out["source"] == "model"
    assert set(out["decisions"]["tools"]) == {"get_weather", "set_mode"}
    assert validate_calls_against_tools(out["function_calls"], tools) == out["function_calls"]
    for call in out["function_calls"]:
        if call["name"] == "set_mode":
            assert call["arguments"]["mode"] in ("a", "b")
            assert isinstance(call["arguments"]["on"], bool)
        if call["name"] == "get_weather":
            assert call["arguments"]["city"] in "what's the weather in Lagos?"  # anchored, not invented
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["stats"]["probes"] == 2 + 1 + 2 + 2  # 2 tools, city, mode + 2 options, on
    ext = agent.extract("Name: Maya Santos. I live in Cebu.", {"name": {"type": "string"}})
    assert set(ext["record"]) <= {"name"}
    assert len(agent.embed("hello")) == model.arc1_config.d_model


def test_schema_memory_caches_engrams():
    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))
    tools = [ToolSpec("get_weather", "Get the weather.", [ToolParam("city")])]
    first = agent.run("weather in Lagos", tools=tools, execute=False)
    second = agent.run("weather in Paris", tools=tools, execute=False)
    assert first["stats"]["schema_encoded"] > 0
    assert second["stats"]["schema_encoded"] == 0
    assert second["stats"]["schema_cached"] == first["stats"]["schema_encoded"] + first["stats"]["schema_cached"]


def test_exclusive_anchoring_resolves_overlaps():
    from gpbacay_arcane.tools import _Probe

    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))  # byte-level: one token per character
    text = "Ada Okonkwo in Lagos"
    utt = agent.codec.utterance(text)
    width, d = len(utt.ids), model.arc1_config.d_model
    name_tok = (1, 11)                     # "Ada Okonkwo"
    city_tok = (1 + text.index("Lagos"), 1 + text.index("Lagos") + 4)

    def peak(span, strength):
        start, end = np.full(width, -5.0), np.full(width, -5.0)
        start[span[0]], end[span[1]] = strength, strength
        return start, end

    rows = [peak(name_tok, 12.0),          # name: confident
            peak(name_tok, 4.0),           # company (optional): same tokens, weaker -> dropped
            peak(name_tok, 3.0)]           # place (required): same tokens, weaker -> re-anchored
    rows[2][0][city_tok[0]] = rows[2][1][city_tok[1]] = 2.5  # runner-up: "Lagos"
    out = {"anchor_start": np.stack([r[0] for r in rows]), "anchor_end": np.stack([r[1] for r in rows]),
           "select_q": np.zeros((3, d)), "select_k": np.zeros((3, d))}
    probes = [_Probe(ROLE_SPAN, "", None, "t", ToolParam("name", required=False)),
              _Probe(ROLE_SPAN, "", None, "t", ToolParam("company", required=False)),
              _Probe(ROLE_SPAN, "", None, "t", ToolParam("place", required=True))]
    got = {x["param"]: x for x in agent._read_arguments(utt, probes, out, np.array([0.99, 0.95, 1.0]))}
    assert got["name"]["present"] and got["name"]["value"] == "Ada Okonkwo"
    assert not got["company"]["present"]
    assert got["place"]["present"] and got["place"]["value"] == "Lagos"


def test_new_schemas_are_encoded_inside_the_same_pass():
    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))
    tools = [ToolSpec("set_mode", "Set mode.", [ToolParam("mode", enum=["eco", "turbo"]), ToolParam("room")])]
    cold = agent.run("turbo mode in the den", tools=tools, execute=False)  # every schema text is new
    warm = agent.run("turbo mode in the den", tools=tools, execute=False)  # every schema text is cached
    assert cold["stats"]["forward_passes"] == warm["stats"]["forward_passes"] == 1
    assert cold["stats"]["schema_encoded"] > 0 and warm["stats"]["schema_encoded"] == 0
    np.testing.assert_allclose(cold["decisions"]["tools"]["set_mode"], warm["decisions"]["tools"]["set_mode"], atol=1e-5)
    # The engram computed inside the joint pass equals a standalone schema encoding.
    text = next(iter(agent.memory._store))
    alone = model.schema_engrams(tf.constant(pad_batch([agent.codec.schema(text)]))).numpy()[0]
    np.testing.assert_allclose(agent.memory.get(text), alone, atol=1e-4)


def test_classify_returns_calibrated_distribution():
    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))
    labels = ["billing", "technical support", "sales"]
    out = agent.classify("my invoice is wrong", labels, task="Route the ticket to a team.")
    assert out["label"] in labels
    assert set(out["distribution"]) == set(labels)
    assert abs(sum(out["distribution"].values()) - 1.0) < 1e-6
    assert out["confidence"] == max(out["distribution"].values())
    assert out["stats"]["forward_passes"] == 1


def test_classification_examples_are_well_formed():
    from gpbacay_arcane.arc1_data import HELD_OUT_TOOLS, sample_classify_example

    lib = build_tool_library()
    rng = random.Random(4)
    for split in ("train", "eval", "unseen_tools"):
        for _ in range(100):
            ex = sample_classify_example(rng, lib, split)
            assert ex.label in ex.labels and len(set(ex.labels)) == len(ex.labels)
            if split == "unseen_tools":
                assert ex.label.replace(" ", "_") in HELD_OUT_TOOLS
    codec = Arc1Codec(BytePairTokenizer(512), 128, 48)
    batch = sample_batch(random.Random(2), codec, lib, n_tool=0, n_extract=0, n_embed=0, n_classify=5)
    assert len(batch["select_param"]) == 5
    assert (batch["select_label"] < (batch["select_options"] >= 0).sum(axis=1)).all()


def test_rcn_roundtrip_header_and_integrity(tmp_path):
    from gpbacay_arcane.rcn import load_rcn, read_header, save_rcn

    model = _small_model()
    tok = BytePairTokenizer(vocab_size=320).train(["weather in Lagos and Paris " * 20])
    utter = tf.constant([[3, 40, 41, 42, 43]])
    a, b, roles = _probe_inputs(model)
    ref = model.decide(utter, tf.zeros((3,), tf.int32), a, b, roles)["fire"].numpy()
    sizes = {}
    for quant, tol in (("f16", 0.05), ("rq8", 0.1), ("rq4", 0.6)):
        path = str(tmp_path / f"m-{quant}.rcn")
        sizes[quant] = save_rcn(model, tok, path, quant=quant, cycles=1)["bytes"]
        head = read_header(path)
        assert head["quant"] == quant and head["cycles"] == 1 and head["d_model"] == model.arc1_config.d_model
        loaded, tok2, _ = load_rcn(path)
        assert loaded.arc1_config.resolve_cycles() == 1          # baked profile
        assert tok2.merges == tok.merges                          # tokenizer travels with the model
        got = loaded.decide(utter, tf.zeros((3,), tf.int32), a, b, roles, cycles=model.arc1_config.binding_cycles)
        np.testing.assert_allclose(got["fire"].numpy(), ref, atol=tol)
    assert sizes["rq4"] < sizes["rq8"] < sizes["f16"]
    raw = bytearray(open(path, "rb").read())
    raw[-1] ^= 0xFF
    bad = str(tmp_path / "bad.rcn")
    open(bad, "wb").write(bytes(raw))
    try:
        load_rcn(bad)
        raise AssertionError("corruption not detected")
    except ValueError as exc:
        assert "corrupted" in str(exc)


def test_coerce_value():
    assert coerce_value("integer", "30%") == (True, 30)
    assert coerce_value("number", "12.50") == (True, 12.5)
    assert coerce_value("integer", "five") == (True, 5)
    assert coerce_value("integer", "Lagos")[0] is False
    assert coerce_value("string", "Lagos") == (True, "Lagos")


def test_parse_agent_json():
    raw = 'prefix {"reasoning":"ok","function_calls":[{"name":"get_weather","arguments":{"city":"Lagos"}}]} junk'
    parsed = parse_agent_json(raw)
    assert parsed["function_calls"][0]["arguments"]["city"] == "Lagos"
    assert parse_agent_json("not json at all")["function_calls"] == []


def test_validate_and_heuristic():
    tools = [
        ToolSpec("get_weather", "weather", [ToolParam(name="city")], handler=lambda city: {"city": city}),
        ToolSpec("set_lights", "lights", [ToolParam(name="room"), ToolParam(name="level", type="integer")]),
    ]
    calls = validate_calls_against_tools(
        [{"name": "get_weather", "arguments": {"city": "Tokyo", "extra": 1}}], tools
    )
    assert calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    assert heuristic_tool_match("what's the weather in Lagos?", tools)[0]["arguments"]["city"] == "Lagos"
    assert heuristic_tool_match("dim the living room to 30", tools)[0]["arguments"]["level"] == 30


def test_arc1_language_model_is_causal():
    tf.random.set_seed(0)
    lm = Arc1LanguageModel(Arc1Config(vocab_size=64, d_model=32, num_layers=2, num_heads=4, seq_len=16,
                                      engram_table_size=128, dropout_rate=0.0))
    x = np.random.RandomState(0).randint(2, 64, (1, 16)).astype(np.int32)
    y = x.copy()
    y[0, 10] = 1
    a, b = lm(x).numpy(), lm(y).numpy()
    assert a.shape == (1, 16, 64)
    np.testing.assert_allclose(a[0, :10], b[0, :10], atol=1e-5)  # the past never sees the future
    assert np.abs(a[0, 10:] - b[0, 10:]).max() > 1e-4
    out = lm.generate([1, 5, 6], max_new_tokens=4, eos_id=None)
    assert len(out) == 7 and all(0 <= t < 64 for t in out)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
