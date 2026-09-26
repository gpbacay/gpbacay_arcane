"""ARC 1 — ARCANE Automation Foundation Model (Resonant Schema Binding).

ARC 1 is an ultra-compact, non-autoregressive "System 1" decision model: it
maps text plus a schema (tools, a record, or a label set) to typed, grounded,
calibrated decisions in one forward pass. It does not generate text.

Pipeline
--------
1. **Perceive once.** The utterance is read a single time by a stack of
   bidirectional ``Arc1PerceptionBlock`` layers (FieldAttention →
   ResonantChannelMixer → ConceptEngram lexical memory → FieldResonance).
   The result is the *utterance field* ``U`` (T x D).
2. **Schema engrams.** Every tool, parameter, enum option, and label is
   perceived by the same blocks and attention-pooled into one vector. Texts
   not yet cached are perceived in the same batch as the utterance
   (``decide_joint``), so a request is always exactly one forward pass.
3. **Resonant binding.** Each engram becomes a *probe* (composed with its
   context engram and a role embedding) that resonates with ``U`` for a few
   shared-weight cycles through a GSER spiking gate (``ResonantBinding``).
   All probes for a request bind in parallel against the same field.
4. **Readouts** on the settled probes:

   * ``fire``   — P(probe fires): a tool applies, an optional argument is
                  present, or a boolean argument is true.
   * ``anchor`` — start/end pointer over utterance tokens, so strings and
                  numbers are copied from the user's words, never invented.
   * ``select`` — an enum argument (or a classification label set) picks the
                  option probe it resonates with most (``q_param . k_option``).
   * embedding  — attention-pooled field, L2-normalised.

Each readout has a temperature fitted on held-out data (``calibration``).
``cycles`` trades accuracy for compute at inference with the same weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import tensorflow as tf

from .arc1_codec import NUM_ROLES
from .layers import Arc1PerceptionBlock
from .mechanisms import RMSNorm, ResonantBinding

_NEG = -1e9

ARC1_PRESETS: Dict[str, Dict] = {
    "arc1-tiny": {
        "vocab_size": 512,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "seq_len": 160,
        "engram_table_size": 4096,
        "engram_rows": 4,
        "binding_heads": 4,
        "binding_cycles": 3,
    },
    "arc1": {
        "vocab_size": 2048,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "seq_len": 256,
        "engram_table_size": 16384,
        "engram_rows": 8,
        "binding_heads": 8,
        "binding_cycles": 4,
    },
}

READOUTS = ("fire", "anchor", "select")


def _default_calibration() -> Dict[str, float]:
    return {name: 1.0 for name in READOUTS}


@dataclass
class Arc1Config:
    """Geometry, binding, and readout calibration for ``Arc1Model``."""

    vocab_size: int = 512
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    seq_len: int = 160
    schema_len: int = 64
    dropout_rate: float = 0.05
    resonance_factor: float = 0.15
    resonance_cycles: int = 3
    spike_threshold: float = 0.4
    leak_rate: float = 0.1
    pad_id: int = 0
    use_rope: bool = True
    gate_normalize: bool = True
    mixer_rank_div: int = 2
    engram_table_size: int = 4096
    engram_rows: int = 4
    ngram_sizes: Tuple[int, ...] = (2, 3)
    binding_heads: int = 4
    binding_cycles: int = 3
    active_cycles: Optional[int] = None  # None = binding_cycles
    calibration: Dict[str, float] = field(default_factory=_default_calibration)

    def resolve_cycles(self, cycles: Optional[int] = None) -> int:
        if cycles is None:
            cycles = self.active_cycles
        if cycles is None:
            return self.binding_cycles
        return int(max(1, min(int(cycles), self.binding_cycles)))

    def temperature(self, readout: str) -> float:
        return float(max((self.calibration or {}).get(readout, 1.0), 1e-3))

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "Arc1Config":
        key = name.lower().replace("_", "-")
        if key not in ARC1_PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {sorted(ARC1_PRESETS)}")
        params = dict(ARC1_PRESETS[key])
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_dict(cls, payload: Dict) -> "Arc1Config":
        """Build from a saved JSON dict, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in dict(payload).items() if k in known}
        if "ngram_sizes" in data:
            data["ngram_sizes"] = tuple(data["ngram_sizes"])
        if "calibration" in data:
            merged = _default_calibration()
            merged.update({k: float(v) for k, v in (data["calibration"] or {}).items() if k in merged})
            data["calibration"] = merged
        return cls(**data)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["ngram_sizes"] = list(self.ngram_sizes)
        data["calibration"] = dict(self.calibration)
        return data

    def with_cycles(self, cycles: Optional[int]) -> "Arc1Config":
        data = self.to_dict()
        data["active_cycles"] = None if cycles is None else int(cycles)
        return Arc1Config.from_dict(data)


class Arc1Model(tf.keras.Model):
    """Resonant Schema Binding model: perceive once, bind every schema probe, read out."""

    def __init__(self, config: Optional[Arc1Config] = None, **kwargs):
        super().__init__(**kwargs)
        self.arc1_config = config or Arc1Config.from_preset("arc1-tiny")
        cfg = self.arc1_config
        if cfg.d_model % cfg.num_heads or cfg.d_model % cfg.binding_heads:
            raise ValueError("d_model must be divisible by num_heads and binding_heads")
        d = cfg.d_model

        self.token_embedding = tf.keras.layers.Embedding(cfg.vocab_size, d, name="token_embedding")
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.blocks = [
            Arc1PerceptionBlock(
                d_model=d,
                num_heads=cfg.num_heads,
                dropout_rate=cfg.dropout_rate,
                resonance_factor=cfg.resonance_factor,
                resonance_cycles=cfg.resonance_cycles,
                spike_threshold=cfg.spike_threshold,
                leak_rate=cfg.leak_rate,
                use_rope=cfg.use_rope,
                max_position=max(cfg.seq_len, cfg.schema_len, 128),
                gate_normalize=cfg.gate_normalize,
                mixer_rank_div=cfg.mixer_rank_div,
                use_engram=(i == 0),  # one lexical memory at the bottom of the stack
                engram_table_size=cfg.engram_table_size,
                engram_rows=cfg.engram_rows,
                ngram_sizes=cfg.ngram_sizes,
                name=f"perception_{i}",
            )
            for i in range(cfg.num_layers)
        ]
        # Attention pooling: one score per token, separately for schemas and utterances.
        self.schema_pool = tf.keras.layers.Dense(1, name="schema_pool")
        self.schema_proj = tf.keras.layers.Dense(d, name="schema_proj")
        self.utter_pool = tf.keras.layers.Dense(1, name="utter_pool")
        self.utter_proj = tf.keras.layers.Dense(d, name="utter_proj")

        self.role_embedding = tf.keras.layers.Embedding(NUM_ROLES, d, name="role_embedding")
        self.compose = tf.keras.layers.Dense(d, name="compose")
        self.compose_norm = RMSNorm(name="compose_norm")
        self.binding = ResonantBinding(
            d_model=d,
            num_heads=cfg.binding_heads,
            cycles=cfg.binding_cycles,
            leak_rate=cfg.leak_rate,
            spike_threshold=cfg.spike_threshold,
            mixer_rank_div=cfg.mixer_rank_div,
            dropout_rate=cfg.dropout_rate,
            name="resonant_binding",
        )
        self.settle_norm = RMSNorm(name="settle_norm")
        self.fire_hidden = tf.keras.layers.Dense(d, activation="gelu", name="fire_hidden")
        self.fire_out = tf.keras.layers.Dense(1, name="fire_out")
        self.anchor_query = tf.keras.layers.Dense(2 * d, name="anchor_query")
        self.anchor_key = tf.keras.layers.Dense(2 * d, name="anchor_key")
        self.select_query = tf.keras.layers.Dense(d, name="select_query")
        self.select_key = tf.keras.layers.Dense(d, name="select_key")

    @classmethod
    def from_preset(cls, name: str = "arc1-tiny", **overrides) -> "Arc1Model":
        return cls(Arc1Config.from_preset(name, **overrides))

    # --------------------------------------------------------------- perceive
    def perceive(self, token_ids, training=False):
        """Utterance / schema field ``(B, T, D)`` and its real-token mask ``(B, T)``."""
        mask = tf.not_equal(token_ids, self.arc1_config.pad_id)
        x = self.embed_dropout(self.token_embedding(token_ids), training=training)
        for block in self.blocks:
            x = block(x, token_ids=token_ids, token_mask=mask, training=training)
        return x, mask

    @staticmethod
    def _attention_pool(field, mask, scorer):
        scores = tf.squeeze(scorer(field), axis=-1)
        scores = tf.where(mask, scores, _NEG)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.reduce_sum(field * tf.expand_dims(weights, -1), axis=1)

    def schema_engrams(self, token_ids, training=False):
        """One engram ``(S, D)`` per schema text (tool, parameter, or option)."""
        field, mask = self.perceive(token_ids, training=training)
        return self.schema_proj(self._attention_pool(field, mask, self.schema_pool))

    def pooled_embedding(self, field, mask):
        vec = self.utter_proj(self._attention_pool(field, mask, self.utter_pool))
        return tf.nn.l2_normalize(vec, axis=-1)

    def embed_text(self, token_ids, training=False):
        field, mask = self.perceive(token_ids, training=training)
        return self.pooled_embedding(field, mask)

    def call(self, token_ids, training=False):
        return self.embed_text(token_ids, training=training)

    # ------------------------------------------------------------------- bind
    def compose_probes(self, engram_a, engram_b, roles):
        """Probe = its own engram + its context engram (tool for a parameter,
        parameter for an option; zeros for a tool) + a role embedding."""
        p = self.compose(tf.concat([engram_a, engram_b], axis=-1)) + self.role_embedding(roles)
        return self.compose_norm(p)

    def bind(self, field, mask, probes, probe_example, cycles=None, training=False):
        """Resonate every probe with its utterance field and read out all heads.

        Returns raw (uncalibrated) logits: ``fire`` (P,), ``anchor_start`` /
        ``anchor_end`` (P, T), plus ``select_q`` / ``select_k`` (P, D).
        """
        cycles = self.arc1_config.resolve_cycles(cycles)
        keys, values = self.binding.field_memory(field)
        settled, heard, peak = self.binding(
            probes, keys, values, mask, probe_example, cycles=cycles, training=training
        )
        settled = self.settle_norm(settled)
        fire = self.fire_out(self.fire_hidden(tf.concat([settled, probes * heard, peak], axis=-1)))

        d = self.arc1_config.d_model
        anchor_keys = tf.gather(self.anchor_key(field), probe_example)  # (P, T, 2D)
        aq = self.anchor_query(settled)
        scale = tf.sqrt(tf.cast(d, field.dtype))
        start = tf.einsum("pd,ptd->pt", aq[:, :d], anchor_keys[..., :d]) / scale
        end = tf.einsum("pd,ptd->pt", aq[:, d:], anchor_keys[..., d:]) / scale
        # Anchors land on real utterance tokens only (never BOS or padding).
        seq = tf.shape(mask)[1]
        not_bos = tf.range(seq) > 0
        anchor_mask = tf.logical_and(tf.gather(mask, probe_example), tf.expand_dims(not_bos, 0))
        return {
            "fire": tf.squeeze(fire, axis=-1),
            "anchor_start": tf.where(anchor_mask, start, _NEG),
            "anchor_end": tf.where(anchor_mask, end, _NEG),
            "select_q": self.select_query(settled),
            "select_k": self.select_key(settled),
        }

    def decide(self, utter_ids, probe_example, engram_a, engram_b, roles, cycles=None, training=False,
               with_embedding=False):
        """One pass: perceive the utterances, compose probes, bind, read out."""
        field, mask = self.perceive(utter_ids, training=training)
        probes = self.compose_probes(engram_a, engram_b, roles)
        out = self.bind(field, mask, probes, probe_example, cycles=cycles, training=training)
        if with_embedding:
            out["embedding"] = self.pooled_embedding(field, mask)
        return out

    def decide_joint(self, token_ids, engram_bank, probe_a, probe_b, roles, cycles=None, training=False):
        """A whole request in exactly one forward pass.

        ``token_ids`` row 0 is the utterance; rows 1.. are schema texts not yet
        cached. All rows share one perception batch (every stage is pad-masked,
        so batching them together changes nothing). New engrams are appended to
        ``engram_bank`` (cached engrams, ``(C, D)``); ``probe_a`` / ``probe_b``
        index the combined bank, with -1 in ``probe_b`` for "no context".
        Returns the readouts plus ``new_engrams`` for the caller to cache.
        """
        field, mask = self.perceive(token_ids, training=training)
        new = self.schema_proj(self._attention_pool(field[1:], mask[1:], self.schema_pool))
        bank = tf.concat([tf.cast(engram_bank, new.dtype), new], axis=0)
        engram_a = tf.gather(bank, probe_a)
        has_b = tf.expand_dims(probe_b >= 0, -1)
        engram_b = tf.where(has_b, tf.gather(bank, tf.maximum(probe_b, 0)), tf.zeros_like(engram_a))
        probes = self.compose_probes(engram_a, engram_b, roles)
        out = self.bind(field[:1], mask[:1], probes, tf.zeros_like(roles), cycles=cycles, training=training)
        out["new_engrams"] = new
        return out

    @staticmethod
    def select_logits(select_q, select_k):
        """``q_param . k_option / sqrt(D)`` — the enum readout."""
        d = tf.cast(tf.shape(select_q)[-1], select_q.dtype)
        return tf.reduce_sum(select_q * select_k, axis=-1) / tf.sqrt(d)

    # ------------------------------------------------------------------ build
    def build_model(self) -> "Arc1Model":
        d = self.arc1_config.d_model
        utter = tf.constant([[3, 40, 41, 42, 43, 0, 0, 0]], dtype=tf.int32)
        schema = tf.constant([[3, 50, 51, 52, 0, 0, 0, 0], [3, 60, 61, 0, 0, 0, 0, 0]], dtype=tf.int32)
        self(utter, training=False)  # Keras 3 marks the model built only via __call__
        eng = self.schema_engrams(schema)
        self.decide(utter, tf.zeros((2,), tf.int32), eng, tf.zeros((2, d)), tf.constant([0, 1]))
        return self

    def get_config(self):
        return self.arc1_config.to_dict()

    @classmethod
    def from_config(cls, config):
        return cls(Arc1Config.from_dict(config))

    def get_model_info(self, cycles: Optional[int] = None) -> Dict:
        built = int(self.count_params()) if self.built else 0
        return {
            "name": "Arc1Model",
            "architecture": "Resonant Schema Binding",
            "built_parameters": built,
            "active_cycles": self.arc1_config.resolve_cycles(cycles),
            "config": self.arc1_config.to_dict(),
        }
