"""ARC 1 — ARCANE Automation Foundation Model (Resonant Schema Binding).

ARC 1 is an ultra-compact, non-autoregressive "System 1" decision model: it
maps text plus a schema (tools, a record, or a label set) to typed, grounded,
calibrated decisions in one bounded computation graph. It does not generate text; the
causal variant ``Arc1LanguageModel`` (bottom of this file) does.

Pipeline
--------
1. **Aligned perception.** The utterance is read by a stack of
   bidirectional ``Arc1PerceptionBlock`` layers (FieldAttention →
   ResonantChannelMixer → ConceptEngram lexical memory → FieldResonance).
   Bounded RSAA cycles project higher states downward, harmonize lower states,
   and re-evaluate the stack before the decision is committed. The result is
   the *utterance field* ``U`` (T x D).
2. **Schema engrams.** Every tool, parameter, enum option, and label is
   perceived by the same blocks and attention-pooled into one vector. Texts
   not yet cached are perceived in the same batch as the utterance
   (``decide_joint``), so alignment remains per example.
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
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from .arc1_codec import NUM_ROLES
from .layers import Arc1HybridBlock, Arc1PerceptionBlock
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
        "rsaa_cycles": 2,
        "active_rsaa_cycles": 0,
    },
    # Causal LM (Arc1LanguageModel) sized for fast CPU distillation: ~10M params, ~40 MB.
    "arc1-lm": {
        "vocab_size": 8000,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "seq_len": 256,
        "engram_table_size": 16384,
        "engram_rows": 8,
        "binding_heads": 8,
        "binding_cycles": 1,
    },
    # Hybrid causal LM: LFM-style local/global scheduling while retaining
    # ARCANE engrams, graded spikes, and selective resonance.  The engram table
    # is deliberately smaller so capacity moves from lexical lookup into depth.
    "arc1-lm-v2": {
        "vocab_size": 8000,
        "d_model": 256,
        "num_layers": 12,
        "num_heads": 8,
        "seq_len": 2048,
        "engram_table_size": 4096,
        "engram_rows": 4,
        "binding_heads": 8,
        "binding_cycles": 1,
        "lm_architecture": "hybrid",
        "lm_block_pattern": ("conv", "conv", "attention") * 4,
        "num_kv_heads": 2,
        "ffn_mult": 1.5,
        "conv_kernel_size": 3,
        "resonance_decays": (0.5, 0.9, 0.99),
        "rope_base": 100000.0,
        "residual_scale": 0.5,
    },
    # Same ARC 1 LM v2 mechanisms, scaled to a practical compact-chat budget.
    # 94,681,648 trainable parameters: 11 local convolution blocks and five
    # global GQA blocks.  A 64-wide attention head keeps the KV cache small.
    "arc1-lm-100m": {
        "vocab_size": 16000,
        "d_model": 640,
        "num_layers": 16,
        "num_heads": 10,
        "seq_len": 4096,
        "engram_table_size": 8192,
        "engram_rows": 4,
        "binding_heads": 10,
        "binding_cycles": 1,
        "lm_architecture": "hybrid",
        "lm_block_pattern": ("conv", "conv", "attention") * 5 + ("conv",),
        "num_kv_heads": 2,
        "ffn_mult": 1.5,
        "conv_kernel_size": 3,
        "resonance_decays": (0.5, 0.9, 0.99),
        "rope_base": 100000.0,
        "residual_scale": 0.5,
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
        "rsaa_cycles": 2,
        "active_rsaa_cycles": 0,
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
    # State-only RSAA. Zero preserves checkpoints saved before ARCANE 5.0.
    rsaa_cycles: int = 0
    active_rsaa_cycles: Optional[int] = None  # None = rsaa_cycles
    rsaa_factor: float = 0.15
    rsaa_epsilon: float = 1e-4
    # ``identity`` reproduces the ARCANE 5.0 state-only implementation and
    # adds no checkpoint variables. ``adaptive`` learns a top-down projection
    # and token-wise update gate for every adjacent pair of perception layers.
    rsaa_mode: str = "identity"
    rsaa_max_factor: float = 0.35
    rsaa_min_improvement: float = 0.0
    rsaa_rollback: bool = True
    calibration: Dict[str, float] = field(default_factory=_default_calibration)
    rsaa_calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Causal-LM-only options.  ``legacy`` preserves every existing checkpoint.
    lm_architecture: str = "legacy"
    lm_block_pattern: Tuple[str, ...] = ()
    num_kv_heads: int = 2
    ffn_mult: float = 1.5
    conv_kernel_size: int = 3
    resonance_decays: Tuple[float, ...] = (0.5, 0.9, 0.99)
    rope_base: float = 10000.0
    residual_scale: float = 1.0

    def resolve_cycles(self, cycles: Optional[int] = None) -> int:
        if cycles is None:
            cycles = self.active_cycles
        if cycles is None:
            return self.binding_cycles
        return int(max(1, min(int(cycles), self.binding_cycles)))

    def resolve_rsaa_cycles(self, cycles: Optional[int] = None) -> int:
        """Resolve a per-call RSAA budget, including the disabled value zero."""
        if cycles is None:
            cycles = self.active_rsaa_cycles
        if cycles is None:
            return self.rsaa_cycles
        return int(max(0, min(int(cycles), self.rsaa_cycles)))

    def temperature(self, readout: str, rsaa_cycles: Optional[int] = None) -> float:
        profile = (self.rsaa_calibration or {}).get(str(self.resolve_rsaa_cycles(rsaa_cycles)), {})
        value = profile.get(readout, (self.calibration or {}).get(readout, 1.0))
        return float(max(value, 1e-3))

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
        if "lm_block_pattern" in data:
            data["lm_block_pattern"] = tuple(data["lm_block_pattern"])
        if "resonance_decays" in data:
            data["resonance_decays"] = tuple(data["resonance_decays"])
        if "calibration" in data:
            merged = _default_calibration()
            merged.update({k: float(v) for k, v in (data["calibration"] or {}).items() if k in merged})
            data["calibration"] = merged
        if "rsaa_calibration" in data:
            profiles = {}
            for profile, values in (data["rsaa_calibration"] or {}).items():
                merged = _default_calibration()
                merged.update({k: float(v) for k, v in (values or {}).items() if k in merged})
                profiles[str(profile)] = merged
            data["rsaa_calibration"] = profiles
        return cls(**data)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["ngram_sizes"] = list(self.ngram_sizes)
        data["lm_block_pattern"] = list(self.lm_block_pattern)
        data["resonance_decays"] = list(self.resonance_decays)
        data["calibration"] = dict(self.calibration)
        data["rsaa_calibration"] = {
            str(profile): dict(values) for profile, values in self.rsaa_calibration.items()
        }
        return data

    def with_cycles(self, cycles: Optional[int]) -> "Arc1Config":
        data = self.to_dict()
        data["active_cycles"] = None if cycles is None else int(cycles)
        return Arc1Config.from_dict(data)


def _perception_blocks(cfg: Arc1Config, causal: bool = False) -> List[Arc1PerceptionBlock]:
    return [
        Arc1PerceptionBlock(
            d_model=cfg.d_model,
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
            causal=causal,
            name=f"perception_{i}",
        )
        for i in range(cfg.num_layers)
    ]


def _hybrid_lm_blocks(cfg: Arc1Config) -> List[Arc1HybridBlock]:
    pattern = tuple(cfg.lm_block_pattern)
    if not pattern:
        pattern = tuple("attention" if (i + 1) % 3 == 0 else "conv" for i in range(cfg.num_layers))
    if len(pattern) != cfg.num_layers:
        raise ValueError("lm_block_pattern length must equal num_layers")
    hidden = max(int(round(cfg.d_model * cfg.ffn_mult)), 16)
    return [
        Arc1HybridBlock(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            sequence_type=kind,
            ffn_hidden_dim=hidden,
            conv_kernel_size=cfg.conv_kernel_size,
            dropout_rate=cfg.dropout_rate,
            spike_threshold=cfg.spike_threshold,
            resonance_decays=cfg.resonance_decays,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            max_position=max(cfg.seq_len, 128),
            use_engram=(i == 0),
            engram_table_size=cfg.engram_table_size,
            engram_rows=cfg.engram_rows,
            ngram_sizes=cfg.ngram_sizes,
            residual_scale=cfg.residual_scale,
            name=f"hybrid_{i}_{kind}",
        )
        for i, kind in enumerate(pattern)
    ]


class Arc1Model(tf.keras.Model):
    """RSAA-aligned schema binding: perceive, align, bind every probe, read out."""

    def __init__(self, config: Optional[Arc1Config] = None, **kwargs):
        super().__init__(**kwargs)
        self.arc1_config = config or Arc1Config.from_preset("arc1-tiny")
        cfg = self.arc1_config
        if cfg.d_model % cfg.num_heads or cfg.d_model % cfg.binding_heads:
            raise ValueError("d_model must be divisible by num_heads and binding_heads")
        if cfg.rsaa_cycles < 0 or not 0.0 < cfg.rsaa_factor <= 1.0 or cfg.rsaa_epsilon < 0.0:
            raise ValueError("RSAA needs cycles >= 0, factor in (0, 1], and epsilon >= 0")
        if cfg.rsaa_mode not in {"identity", "adaptive"}:
            raise ValueError("rsaa_mode must be 'identity' or 'adaptive'")
        if not cfg.rsaa_factor <= cfg.rsaa_max_factor <= 1.0:
            raise ValueError("rsaa_max_factor must be between rsaa_factor and 1")
        if cfg.rsaa_min_improvement < 0.0:
            raise ValueError("rsaa_min_improvement must be non-negative")
        d = cfg.d_model

        self.token_embedding = tf.keras.layers.Embedding(cfg.vocab_size, d, name="token_embedding")
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.blocks = _perception_blocks(cfg)
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

        # Register optional variables after every legacy layer. Keras' weight
        # object graph then remains compatible with pre-RSAA checkpoints when
        # ``load_weights(..., skip_mismatch=True)`` warm-starts this model.
        self.rsaa_projections = []
        self.rsaa_gates = []
        if cfg.rsaa_mode == "adaptive":
            fraction = (cfg.rsaa_factor - 1e-3) / max(cfg.rsaa_max_factor - 1e-3, 1e-6)
            fraction = min(max(fraction, 1e-4), 1.0 - 1e-4)
            gate_bias = float(tf.math.log(fraction / (1.0 - fraction)))
            for i in range(max(cfg.num_layers - 1, 0)):
                self.rsaa_projections.append(
                    tf.keras.layers.Dense(
                        d,
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.Identity(),
                        name=f"rsaa_feedback_{i}",
                    )
                )
                self.rsaa_gates.append(
                    tf.keras.layers.Dense(
                        1,
                        kernel_initializer="zeros",
                        bias_initializer=tf.keras.initializers.Constant(gate_bias),
                        name=f"rsaa_gate_{i}",
                    )
                )

    @classmethod
    def from_preset(cls, name: str = "arc1-tiny", **overrides) -> "Arc1Model":
        return cls(Arc1Config.from_preset(name, **overrides))

    # --------------------------------------------------------------- perceive
    def _rsaa_target(self, boundary, upper):
        if self.arc1_config.rsaa_mode == "adaptive":
            return self.rsaa_projections[boundary](upper)
        return upper

    def _rsaa_divergence(self, states, mask):
        """Per-example mean squared disagreement between adjacent hierarchy states."""
        if len(states) < 2:
            return tf.zeros((tf.shape(mask)[0],), dtype=states[0].dtype)
        weight = tf.cast(tf.expand_dims(mask, -1), states[0].dtype)
        width = tf.cast(tf.shape(states[0])[-1], states[0].dtype)
        denom = tf.maximum(tf.reduce_sum(weight, axis=[1, 2]) * width, 1.0)
        terms = []
        for i, (lower, upper) in enumerate(zip(states[:-1], states[1:])):
            target = self._rsaa_target(i, upper)
            terms.append(tf.reduce_sum(tf.square(lower - target) * weight, axis=[1, 2]) / denom)
        return tf.add_n(terms)

    def _rsaa_update(self, boundary, lower, upper):
        """Project one upper state down and return its gated lower-state update."""
        cfg = self.arc1_config
        target = self._rsaa_target(boundary, upper)
        delta = target - lower
        if cfg.rsaa_mode == "adaptive":
            features = tf.concat([lower, target, tf.abs(delta)], axis=-1)
            gate = tf.sigmoid(self.rsaa_gates[boundary](features))
            lo = tf.cast(1e-3, lower.dtype)
            hi = tf.cast(cfg.rsaa_max_factor, lower.dtype)
            gamma = lo + (hi - lo) * tf.cast(gate, lower.dtype)
        else:
            gamma = tf.cast(cfg.rsaa_factor, lower.dtype)
        return lower + gamma * delta, gamma

    def perceive(self, token_ids, training=False, rsaa_cycles=None, return_rsaa=False):
        """Return a field and mask after optional state-only RSAA alignment.

        Identity mode preserves the checkpoint-compatible ARCANE 5.0 update.
        Adaptive mode uses learned layer-specific top-down projections and
        token-wise gates. At inference, a cycle is committed per example only
        when it reduces hierarchical divergence; a harmful cycle is rolled
        back without consuming later cycles.
        """
        mask = tf.not_equal(token_ids, self.arc1_config.pad_id)
        embedded = self.embed_dropout(self.token_embedding(token_ids), training=training)
        x = embedded
        states = []
        for block in self.blocks:
            x = block(x, token_ids=token_ids, token_mask=mask, training=training)
            states.append(x)

        budget = self.arc1_config.resolve_rsaa_cycles(rsaa_cycles)
        divergence = self._rsaa_divergence(states, mask)
        initial_divergence = divergence
        steps = tf.zeros_like(divergence, dtype=tf.int32)
        epsilon = tf.cast(self.arc1_config.rsaa_epsilon, divergence.dtype)
        accepted_steps = tf.zeros_like(divergence, dtype=tf.int32)
        rolled_back = tf.zeros_like(divergence, dtype=tf.int32)
        gate_total = tf.zeros_like(divergence)

        for _ in range(budget):
            active = divergence >= epsilon
            targets = states[1:]
            x = embedded
            candidates = []
            for i, block in enumerate(self.blocks):
                x = block(x, token_ids=token_ids, token_mask=mask, training=training)
                if i < len(self.blocks) - 1:
                    aligned, gamma = self._rsaa_update(i, x, targets[i])
                    align_here = tf.logical_and(mask, tf.expand_dims(active, -1))
                    x = tf.where(tf.expand_dims(align_here, -1), aligned, x)
                    gate_value = tf.cast(gamma, divergence.dtype)
                    if len(gate_value.shape) == 0:
                        gate_value = tf.fill(tf.shape(x)[:2], gate_value)
                    else:
                        gate_value = tf.squeeze(gate_value, axis=-1)
                    mean_gate = tf.math.divide_no_nan(
                        tf.reduce_sum(gate_value * tf.cast(mask, gate_value.dtype), axis=1),
                        tf.reduce_sum(tf.cast(mask, gate_value.dtype), axis=1),
                    )
                    gate_total += tf.where(active, mean_gate, tf.zeros_like(mean_gate))
                candidates.append(x)
            candidate_divergence = self._rsaa_divergence(candidates, mask)
            if self.arc1_config.rsaa_rollback and not training:
                improvement = divergence - candidate_divergence
                accept = tf.logical_and(
                    active,
                    improvement >= tf.cast(self.arc1_config.rsaa_min_improvement, improvement.dtype),
                )
            else:
                accept = active
            choose = tf.reshape(accept, (-1, 1, 1))
            states = [tf.where(choose, new, old) for new, old in zip(candidates, states)]
            divergence = tf.where(accept, candidate_divergence, divergence)
            steps += tf.cast(active, tf.int32)
            accepted_steps += tf.cast(accept, tf.int32)
            rolled_back += tf.cast(tf.logical_and(active, tf.logical_not(accept)), tf.int32)

        trace = {
            "rsaa_initial_divergence": initial_divergence,
            "rsaa_divergence": divergence,
            "rsaa_steps": steps,
            "rsaa_accepted_steps": accepted_steps,
            "rsaa_rolled_back": rolled_back,
            "rsaa_mean_factor": tf.math.divide_no_nan(
                gate_total,
                tf.cast(tf.maximum(steps, 1) * max(len(self.blocks) - 1, 1), gate_total.dtype),
            ),
        }
        if return_rsaa:
            return states[-1], mask, trace
        return states[-1], mask

    @staticmethod
    def _attention_pool(field, mask, scorer):
        scores = tf.squeeze(scorer(field), axis=-1)
        scores = tf.where(mask, scores, _NEG)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.reduce_sum(field * tf.expand_dims(weights, -1), axis=1)

    def schema_engrams(self, token_ids, training=False, rsaa_cycles=None):
        """One engram ``(S, D)`` per schema text (tool, parameter, or option)."""
        field, mask = self.perceive(token_ids, training=training, rsaa_cycles=rsaa_cycles)
        return self.schema_proj(self._attention_pool(field, mask, self.schema_pool))

    def pooled_embedding(self, field, mask):
        vec = self.utter_proj(self._attention_pool(field, mask, self.utter_pool))
        return tf.nn.l2_normalize(vec, axis=-1)

    def embed_text(self, token_ids, training=False, rsaa_cycles=None):
        field, mask = self.perceive(token_ids, training=training, rsaa_cycles=rsaa_cycles)
        return self.pooled_embedding(field, mask)

    def call(self, token_ids, training=False):
        return self.embed_text(token_ids, training=training)

    # ------------------------------------------------------------------- bind
    def compose_probes(self, engram_a, engram_b, roles):
        """Probe = its own engram + its context engram (tool for a parameter,
        parameter for an option; zeros for a tool) + a role embedding."""
        p = self.compose(tf.concat([engram_a, engram_b], axis=-1)) + self.role_embedding(roles)
        return self.compose_norm(p)

    def bind(self, field, mask, probes, probe_example, cycles=None, training=False, dynamic=None):
        """Resonate every probe with its utterance field and read out all heads.

        Returns raw (uncalibrated) logits: ``fire`` (P,), ``anchor_start`` /
        ``anchor_end`` (P, T), plus ``select_q`` / ``select_k`` (P, D).
        ``binding_steps`` is how many cycles each probe applied. Omitting
        ``cycles`` at inference stops a probe once its update settles.
        """
        if dynamic is None:
            dynamic = cycles is None and not training
        cycles = self.arc1_config.resolve_cycles(cycles)
        keys, values = self.binding.field_memory(field)
        settled, heard, peak, binding_steps = self.binding(
            probes, keys, values, mask, probe_example, cycles=cycles, training=training,
            dynamic=bool(dynamic) and not training,
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
            "binding_steps": binding_steps,
        }

    def decide(self, utter_ids, probe_example, engram_a, engram_b, roles, cycles=None, training=False,
               with_embedding=False, rsaa_cycles=None, dynamic_binding=None):
        """Align perception, compose probes, bind, and read out without decoding."""
        field, mask, trace = self.perceive(
            utter_ids, training=training, rsaa_cycles=rsaa_cycles, return_rsaa=True
        )
        probes = self.compose_probes(engram_a, engram_b, roles)
        out = self.bind(
            field, mask, probes, probe_example, cycles=cycles, training=training, dynamic=dynamic_binding,
        )
        out.update(trace)
        if with_embedding:
            out["embedding"] = self.pooled_embedding(field, mask)
        return out

    def decide_joint(self, token_ids, engram_bank, probe_a, probe_b, roles, cycles=None, training=False,
                     rsaa_cycles=None, dynamic_binding=None):
        """A whole request in one bounded, non-autoregressive computation graph.

        ``token_ids`` row 0 is the utterance; rows 1.. are schema texts not yet
        cached. All rows share one perception batch (every stage is pad-masked,
        so batching them together changes nothing). New engrams are appended to
        ``engram_bank`` (cached engrams, ``(C, D)``); ``probe_a`` / ``probe_b``
        index the combined bank, with -1 in ``probe_b`` for "no context".
        Returns the readouts plus ``new_engrams`` for the caller to cache.
        """
        field, mask, trace = self.perceive(
            token_ids, training=training, rsaa_cycles=rsaa_cycles, return_rsaa=True
        )
        new = self.schema_proj(self._attention_pool(field[1:], mask[1:], self.schema_pool))
        bank = tf.concat([tf.cast(engram_bank, new.dtype), new], axis=0)
        engram_a = tf.gather(bank, probe_a)
        has_b = tf.expand_dims(probe_b >= 0, -1)
        engram_b = tf.where(has_b, tf.gather(bank, tf.maximum(probe_b, 0)), tf.zeros_like(engram_a))
        probes = self.compose_probes(engram_a, engram_b, roles)
        out = self.bind(
            field[:1], mask[:1], probes, tf.zeros_like(roles), cycles=cycles, training=training,
            dynamic=dynamic_binding,
        )
        out["new_engrams"] = new
        out.update(trace)
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
        if self.arc1_config.rsaa_mode == "adaptive" and self.arc1_config.rsaa_cycles:
            self.perceive(utter, training=False, rsaa_cycles=1)
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
            "architecture": "RSAA-Capable Resonant Schema Binding",
            "built_parameters": built,
            "active_cycles": self.arc1_config.resolve_cycles(cycles),
            "active_rsaa_cycles": self.arc1_config.resolve_rsaa_cycles(),
            "rsaa_cycles": self.arc1_config.rsaa_cycles,
            "rsaa_mode": self.arc1_config.rsaa_mode,
            "config": self.arc1_config.to_dict(),
        }


class Arc1LanguageModel(tf.keras.Model):
    """ARC 1 as a small causal language model (the docs chat model).

    Same perception stack as ``Arc1Model`` (FieldAttention -> ResonantChannelMixer
    -> ConceptEngram -> FieldResonance) run in causal mode, plus a tied
    embedding / LM head. Exposes ``slm_config`` and ``generate`` so it is a
    drop-in for ``ArcaneSmallLanguageModel`` in distillation and serving.
    """

    def __init__(self, config: Optional[Arc1Config] = None, **kwargs):
        super().__init__(**kwargs)
        self.slm_config = config or Arc1Config.from_preset("arc1-lm")
        cfg = self.slm_config
        self.token_embedding = tf.keras.layers.Embedding(cfg.vocab_size, cfg.d_model, name="token_embedding")
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.blocks = (
            _hybrid_lm_blocks(cfg)
            if cfg.lm_architecture == "hybrid"
            else _perception_blocks(cfg, causal=True)
        )
        self.final_norm = RMSNorm(name="final_norm")

    @classmethod
    def from_preset(cls, name: str = "arc1-lm", **overrides) -> "Arc1LanguageModel":
        return cls(Arc1Config.from_preset(name, **overrides))

    def call(self, token_ids, training=False):
        mask = tf.not_equal(token_ids, self.slm_config.pad_id)
        x = self.embed_dropout(self.token_embedding(token_ids), training=training)
        for block in self.blocks:
            x = block(x, token_ids=token_ids, token_mask=mask, training=training)
        return tf.matmul(self.final_norm(x), self.token_embedding.embeddings, transpose_b=True)

    def prefill(self, token_ids):
        """Run a hybrid prompt once and return logits plus incremental state.

        This API is intentionally limited to the v2 hybrid architecture; legacy
        checkpoints retain their exact historical execution path.
        """
        if self.slm_config.lm_architecture != "hybrid":
            raise ValueError("prefill is available only for the hybrid ARC 1 LM")
        token_ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        mask = tf.not_equal(token_ids, self.slm_config.pad_id)
        x = self.token_embedding(token_ids)
        states = []
        for block in self.blocks:
            x, state = block.call_with_cache(
                x, token_ids=token_ids, token_mask=mask, training=False)
            states.append(state)
        logits = tf.matmul(self.final_norm(x), self.token_embedding.embeddings, transpose_b=True)
        keep = max(self.slm_config.ngram_sizes, default=1) - 1
        recent = token_ids[:, -keep:] if keep else token_ids[:, :0]
        return logits, {
            "blocks": states,
            "position": tf.shape(token_ids)[1],
            "recent_token_ids": recent,
        }

    @tf.function(reduce_retracing=True)
    def decode_step(self, token_ids, state):
        """Decode one new token using state returned by :meth:`prefill`."""
        if self.slm_config.lm_architecture != "hybrid":
            raise ValueError("decode_step is available only for the hybrid ARC 1 LM")
        token_ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        if len(token_ids.shape) == 1:
            token_ids = tf.expand_dims(token_ids, 1)
        recent = tf.concat([state["recent_token_ids"], token_ids], axis=1)
        max_ngram = max(self.slm_config.ngram_sizes, default=1)
        recent = recent[:, -max_ngram:]
        x = self.token_embedding(token_ids)
        next_states = []
        for block, block_state in zip(self.blocks, state["blocks"]):
            x, next_state = block.step(
                x, block_state, position=state["position"], recent_token_ids=recent,
                training=False)
            next_states.append(next_state)
        logits = tf.matmul(self.final_norm(x), self.token_embedding.embeddings, transpose_b=True)
        keep = max_ngram - 1
        return logits, {
            "blocks": next_states,
            "position": state["position"] + 1,
            "recent_token_ids": recent[:, -keep:] if keep else recent[:, :0],
        }

    def build_model(self) -> "Arc1LanguageModel":
        # Layers build dynamically in sequence length; avoid an unnecessary
        # quadratic full-context pass merely to create their variables.
        self(tf.ones((1, min(self.slm_config.seq_len, 8)), dtype=tf.int32), training=False)
        return self

    def generate(self, token_ids, max_new_tokens=50, temperature=0.3, top_k=40, eos_id=1,
                 allowed_token_ids=None, repetition_penalty=1.08,
                 no_repeat_ngram_size=3) -> List[int]:
        from .language_model import _sample_logits

        cfg = self.slm_config
        tokens = [int(t) for t in token_ids] or [1]
        if cfg.lm_architecture == "hybrid":
            window = tokens[-cfg.seq_len:]
            logits, state = self.prefill(tf.constant([window], dtype=tf.int32))
            next_logits = logits[0, -1]
            for _ in range(max_new_tokens):
                next_id = _sample_logits(
                    next_logits, temperature=temperature, top_k=top_k,
                    allowed_ids=allowed_token_ids, previous_ids=tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                tokens.append(int(next_id))
                if eos_id is not None and next_id == eos_id:
                    break
                if int(state["position"].numpy()) >= cfg.seq_len:
                    break
                step_logits, state = self.decode_step(tf.constant([next_id]), state)
                next_logits = step_logits[0, -1]
            return tokens
        # ponytail: full-window recompute per token, add a KV cache if chat latency matters
        for _ in range(max_new_tokens):
            window = tokens[-cfg.seq_len:]
            logits = self(tf.constant([window], dtype=tf.int32), training=False)[0, -1]
            next_id = _sample_logits(
                logits, temperature=temperature, top_k=top_k, allowed_ids=allowed_token_ids,
                previous_ids=tokens, repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            tokens.append(int(next_id))
            if eos_id is not None and next_id == eos_id:
                break
        return tokens

    def get_config(self):
        return self.slm_config.to_dict()

    @classmethod
    def from_config(cls, config):
        return cls(Arc1Config.from_dict(config))
