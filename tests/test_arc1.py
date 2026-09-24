"""Tests for ARC 1 layers, ladder, decision heads, codec, data, and agent."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.arc1 import Arc1Config, Arc1Model, nested_ladder_indices
from gpbacay_arcane.arc1_codec import Arc1Codec, best_span, pad_batch
from gpbacay_arcane.arc1_data import build_tool_library, sample_extract_example, sample_tool_example
from gpbacay_arcane.arc1_train import BATCH_SIGNATURE, compute_losses, sample_batch
from gpbacay_arcane.layers import Arc1DecoderBlock, ResonantChannelMixer
from gpbacay_arcane.mechanisms import ConceptEngram
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

SMALL = dict(vocab_size=512, d_model=32, num_layers=2, num_heads=2, seq_len=256,
             engram_table_size=256, engram_rows=2, dropout_rate=0.0, ladder_depths=(1, 2), softmax_every=2)


def _small_model():
    model = Arc1Model(Arc1Config(**SMALL))
    model.build_model()
    return model


def test_nested_ladder_nests_and_spreads():
    s2 = nested_ladder_indices(12, 2)
    s4 = nested_ladder_indices(12, 4)
    s12 = nested_ladder_indices(12, 12)
    assert s2 == [0, 11]
    assert set(s2).issubset(s4)
    assert set(s4).issubset(s12)
    assert s12 == list(range(12))
    assert max(s4) == 11 and min(s4) == 0


def test_concept_engram_forward_shape():
    layer = ConceptEngram(d_model=32, table_size=128, rows_per_token=4, ngram_sizes=(2, 3))
    x = tf.random.normal((2, 8, 32))
    ids = tf.constant(np.random.randint(0, 50, size=(2, 8)), dtype=tf.int32)
    assert layer(x, token_ids=ids).shape == (2, 8, 32)


def test_concept_engram_rows_are_independent_hashes():
    layer = ConceptEngram(d_model=8, table_size=4096, rows_per_token=4, ngram_sizes=(2, 3))
    ids = tf.constant(np.random.RandomState(0).randint(4, 260, size=(1, 64)), dtype=tf.int32)
    keys = layer._lookup_keys(ids).numpy()[0]  # (T, R)
    assert keys.min() >= 0 and keys.max() < 4096
    # Old scheme used base+i (adjacent slots); independent hashes should not.
    gaps = np.abs(np.diff(keys, axis=-1))
    assert np.mean(gaps <= 3) < 0.05


def test_resonant_channel_mixer_forward():
    mix = ResonantChannelMixer(d_model=32, rank_div=4)
    assert mix(tf.random.normal((2, 8, 32))).shape == (2, 8, 32)


def test_arc1_decoder_block_smoke():
    block = Arc1DecoderBlock(d_model=32, num_heads=4, engram_table_size=64, engram_rows=2,
                             dropout_rate=0.0, use_rope=True, use_decay=False, chunk_size=32)
    y = block(tf.random.normal((1, 8, 32)), token_ids=tf.zeros((1, 8), dtype=tf.int32), training=False)
    assert y.shape == (1, 8, 32)


def test_arc1_heads_shapes_and_embedding_norm():
    model = _small_model()
    ids = tf.constant(pad_batch([[3, 10, 11, 12], [3, 20, 21]]))
    out = model.decide(ids, with_lm=True)
    assert out["noul"].shape == (2,)
    assert out["choice"].shape == (2,)
    assert out["span_start"].shape == ids.shape
    assert out["lm"].shape[-1] == model.arc1_config.vocab_size
    assert model.confidence(ids).shape == (2,)
    emb = model.embed_text(ids)
    assert emb.shape == (2, model.arc1_config.d_model)
    np.testing.assert_allclose(tf.norm(emb, axis=-1).numpy(), 1.0, atol=1e-4)


def test_decisions_ignore_right_padding():
    model = _small_model()
    seq = [3, 40, 41, 42, 43, 44]
    short = model.decide(tf.constant([seq]))
    padded = model.decide(tf.constant([seq + [0] * 26]))
    np.testing.assert_allclose(short["noul"].numpy(), padded["noul"].numpy(), atol=1e-4)
    e1 = model.embed_text(tf.constant([seq])).numpy()
    e2 = model.embed_text(tf.constant([seq + [0] * 26])).numpy()
    np.testing.assert_allclose(e1, e2, atol=1e-4)


def test_ladder_depth_is_per_call_and_stateless():
    model = _small_model()
    ids = tf.constant([[3, 5, 6, 7]])
    assert model.arc1_config.ladder_block_indices(1) == [0]
    full = model.decide(ids)["noul"].numpy()
    shallow = model.decide(ids, depth=1)["noul"].numpy()
    assert not np.allclose(full, shallow)
    assert model.arc1_config.resolve_depth() == model.arc1_config.num_layers  # not mutated
    model.select_ladder_depth(1)
    assert model.arc1_config.resolve_depth() == 1


def test_config_roundtrip_keeps_calibration():
    cfg = Arc1Config(**SMALL)
    cfg.calibration = {"noul": 1.7, "choice": 2.0, "span": 0.9}
    again = Arc1Config.from_dict(cfg.to_dict())
    assert again.calibration == cfg.calibration
    assert again.ladder_depths == (1, 2)
    assert Arc1Config.from_dict({"d_model": 32, "unknown_key": 1}).d_model == 32


def test_codec_span_roundtrip_with_bpe():
    tok = BytePairTokenizer(vocab_size=320).train(["weather in San Francisco and weather in Lagos " * 20])
    codec = Arc1Codec(tok, 256)
    user = "what's the weather in San Francisco today?"
    a = user.index("San Francisco")
    seq = codec.arg_seq("get_weather", "city", "string", "City name", True, user)
    s, e = codec.char_span_to_tokens(seq, user, (a, a + len("San Francisco")))
    assert seq.span_lo <= s <= e < seq.span_hi
    assert codec.tokens_to_text(seq, s, e) == "San Francisco"


def test_best_span_respects_order():
    start = np.array([0, 5, 0, 0], dtype=np.float32)
    end = np.array([9, 0, 0, 6], dtype=np.float32)  # end 0 is before start 1
    s, e, p = best_span(start, end, 0, 4)
    assert s <= e and 0 < p <= 1


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
    # Held-out evaluation never sees synthesized values: every city comes from the real eval pool.
    lib = build_tool_library()
    for ex in fixed_eval_set("eval", 200):
        for call in ex.calls:
            if call.tool == "get_weather":
                assert call.arguments["city"].lower() in {c.lower() for c in CITIES["eval"]}
    assert all(t.name in lib for ex in fixed_eval_set("eval", 50) for t in ex.tools)


def test_procedural_tools_have_grounded_spans():
    from gpbacay_arcane.arc1_data import procedural_library, _single

    rng = random.Random(5)
    for _ in range(40):
        for tool in procedural_library(rng, 4).values():
            text, call = _single(rng, tool, "train")
            for name, (a, b) in call.spans.items():
                value = call.arguments[name]
                assert text[a:b] == value if isinstance(value, str) else coerce_value("number", text[a:b])[0]


def test_training_step_reduces_loss_on_fixed_batch():
    model = _small_model()
    codec = Arc1Codec(BytePairTokenizer(512), model.arc1_config.seq_len)
    batch = sample_batch(random.Random(0), codec, build_tool_library(), n_tool=3, n_extract=1, n_embed=3)
    batch = {k: tf.constant(v) for k, v in batch.items()}
    opt = tf.keras.optimizers.Adam(3e-3)
    opt.build(model.trainable_variables)

    @tf.function(input_signature=[BATCH_SIGNATURE])
    def step(b):
        with tf.GradientTape() as tape:
            losses, _ = compute_losses(model, b, depth=2, training=True)
        grads = tape.gradient(losses["total"], model.trainable_variables)
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return losses["total"]

    first = float(step(batch))
    for _ in range(25):
        last = float(step(batch))
    assert last < first * 0.7, (first, last)


def test_agent_output_is_schema_valid_even_untrained():
    model = _small_model()
    agent = Arc1Agent(model, BytePairTokenizer(512))
    tools = [
        ToolSpec("get_weather", "Get the weather.", [ToolParam("city")], handler=lambda city: {"city": city}),
        ToolSpec("set_mode", "Set mode.", [ToolParam("mode", enum=["a", "b"]), ToolParam("on", type="boolean")]),
    ]
    agent.tool_threshold = 0.0  # force every tool through argument filling
    out = agent.run("what's the weather in Lagos?", tools=tools, execute=False, depth=1)
    assert out["source"] == "model"
    assert set(out["decisions"]["tools"]) == {"get_weather", "set_mode"}
    assert validate_calls_against_tools(out["function_calls"], tools) == out["function_calls"]
    for call in out["function_calls"]:
        if call["name"] == "set_mode":
            assert call["arguments"]["mode"] in ("a", "b")
            assert isinstance(call["arguments"]["on"], bool)
    assert 0.0 <= out["confidence"] <= 1.0
    ext = agent.extract("Name: Maya Santos. I live in Cebu.", {"name": {"type": "string"}})
    assert set(ext["record"]) <= {"name"}
    assert len(agent.embed("hello")) == model.arc1_config.d_model


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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
