"""ARC 1 — ARCANE Automation Foundation Model.

A laddered ARCANE decoder (ResonantChannelMixer, ConceptEngram,
ResonantSequenceMixer, causal linear/softmax attention) with Laya-style typed
decision heads on top instead of free-form JSON generation:

* ``noul``   — calibrated P(true) for a yes/no question read at the last token
              (does this tool apply? is this argument present? boolean value).
* ``choice`` — one logit per candidate sequence; softmax over a candidate group
              (enum values, record labels).
* ``span``   — start/end pointer over the user-text tokens, so string/number
              arguments are copied from the input and never invented.

Every head has a scalar temperature fit on held-out data (``calibration``),
so reported probabilities are calibrated. The LM head is kept for generation
and as an auxiliary training signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import tensorflow as tf

from .layers import Arc1DecoderBlock
from .mechanisms import AttentionResidual, RMSNorm
from .language_model import _sample_logits


def nested_ladder_indices(num_layers: int, depth: int) -> List[int]:
    """Needle-style nested midpoint block sets.

    ``S_2 = {0, L-1}``, then repeatedly insert the midpoint of the widest gap
    until ``|S_d| = depth``. Sets nest: ``S_2 ⊂ S_3 ⊂ … ⊂ S_L``.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    depth = int(max(1, min(depth, num_layers)))
    if depth == 1:
        return [0]
    selected = {0, num_layers - 1}
    while len(selected) < depth:
        ordered = sorted(selected)
        best_gap = -1
        best_mid = None
        for a, b in zip(ordered, ordered[1:]):
            gap = b - a
            if gap > best_gap and gap > 1:
                best_gap = gap
                best_mid = (a + b) // 2
        if best_mid is None or best_mid in selected:
            # Fill any remaining holes left-to-right.
            for i in range(num_layers):
                if i not in selected:
                    selected.add(i)
                    break
        else:
            selected.add(best_mid)
    return sorted(selected)


ARC1_PRESETS: Dict[str, Dict] = {
    "arc1-tiny": {
        "vocab_size": 512,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "seq_len": 384,
        "dropout_rate": 0.05,
        "engram_table_size": 4096,
        "engram_rows": 4,
        "mixer_rank_div": 2,
        "ladder_depths": (2, 4),
        "softmax_every": 2,
    },
    "arc1": {
        "vocab_size": 8000,
        "d_model": 256,
        "num_layers": 12,
        "num_heads": 8,
        "seq_len": 512,
        "dropout_rate": 0.05,
        "engram_table_size": 16384,
        "engram_rows": 8,
        "mixer_rank_div": 4,
        "ladder_depths": (2, 4, 6, 8, 12),
        "softmax_every": 4,
    },
}

HEAD_NAMES = ("noul", "choice", "span")


def _default_calibration() -> Dict[str, float]:
    return {name: 1.0 for name in HEAD_NAMES}


@dataclass
class Arc1Config:
    """Width/depth/engram geometry and head calibration for ``Arc1Model``."""

    vocab_size: int = 8000
    d_model: int = 256
    num_layers: int = 12
    num_heads: int = 8
    seq_len: int = 512
    dropout_rate: float = 0.05
    resonance_factor: float = 0.15
    resonance_cycles: int = 3
    spike_threshold: float = 0.4
    leak_rate: float = 0.1
    pad_id: int = 0
    softmax_every: int = 0
    use_rope: bool = True
    use_decay: bool = True
    chunk_size: int = 64
    reweight_centered: bool = True
    gate_normalize: bool = True
    norm_type: str = "rms"
    mixer_rank_div: int = 4
    engram_table_size: int = 16384
    engram_rows: int = 8
    ngram_sizes: Tuple[int, ...] = (2, 3)
    ladder_depths: Tuple[int, ...] = (2, 4, 6, 8, 12)
    active_depth: Optional[int] = None  # None = full num_layers
    calibration: Dict[str, float] = field(default_factory=_default_calibration)

    def attention_types(self) -> List[str]:
        if not self.softmax_every:
            return ["linear"] * self.num_layers
        return [
            "softmax" if (i + 1) % self.softmax_every == 0 else "linear"
            for i in range(self.num_layers)
        ]

    def resolve_depth(self, depth: Optional[int] = None) -> int:
        if depth is None:
            depth = self.active_depth
        if depth is None:
            return self.num_layers
        return int(max(1, min(depth, self.num_layers)))

    def ladder_block_indices(self, depth: Optional[int] = None) -> List[int]:
        return nested_ladder_indices(self.num_layers, self.resolve_depth(depth))

    def temperature(self, head: str) -> float:
        return float(max((self.calibration or {}).get(head, 1.0), 1e-3))

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
        for key in ("ngram_sizes", "ladder_depths"):
            if key in data:
                data[key] = tuple(data[key])
        if "calibration" in data:
            merged = _default_calibration()
            merged.update({k: float(v) for k, v in (data["calibration"] or {}).items()})
            data["calibration"] = merged
        return cls(**data)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["ngram_sizes"] = list(self.ngram_sizes)
        data["ladder_depths"] = list(self.ladder_depths)
        data["calibration"] = dict(self.calibration)
        return data

    def with_depth(self, depth: Optional[int]) -> "Arc1Config":
        data = self.to_dict()
        data["active_depth"] = None if depth is None else int(depth)
        return Arc1Config.from_dict(data)


def _head_mlp(d_model: int, units: int, name: str) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_model, activation="gelu", name=f"{name}_hidden"),
            tf.keras.layers.Dense(units, name=f"{name}_out"),
        ],
        name=name,
    )


class Arc1Model(tf.keras.Model):
    """Decoder-only ARC 1 with LM, decision, span, and embedding heads.

    Inputs are right-padded with ``pad_id``; every head reads real tokens only.
    ``depth`` selects a nested ladder slice per call without mutating state.
    """

    def __init__(self, config: Optional[Arc1Config] = None, **kwargs):
        super().__init__(**kwargs)
        self.arc1_config = config or Arc1Config.from_preset("arc1-tiny")
        cfg = self.arc1_config
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.token_embedding = tf.keras.layers.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            name="token_embedding",
        )
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.depth_residuals = [
            AttentionResidual(cfg.d_model, name=f"attnres_{i}")
            for i in range(cfg.num_layers)
        ]
        self.blocks = [
            Arc1DecoderBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                dropout_rate=cfg.dropout_rate,
                resonance_factor=cfg.resonance_factor,
                resonance_cycles=cfg.resonance_cycles,
                spike_threshold=cfg.spike_threshold,
                leak_rate=cfg.leak_rate,
                attention_type=kind,
                use_rope=cfg.use_rope,
                max_position=max(cfg.seq_len, 2048),
                chunk_size=cfg.chunk_size,
                use_decay=cfg.use_decay,
                reweight_centered=cfg.reweight_centered,
                gate_normalize=cfg.gate_normalize,
                norm_type=cfg.norm_type,
                mixer_rank_div=cfg.mixer_rank_div,
                engram_table_size=cfg.engram_table_size,
                engram_rows=cfg.engram_rows,
                ngram_sizes=cfg.ngram_sizes,
                name=f"arc1_block_{i}",
            )
            for i, kind in enumerate(cfg.attention_types())
        ]
        self.final_norm = (
            RMSNorm(eps=1e-6, name="final_norm")
            if cfg.norm_type == "rms"
            else tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        )
        self.noul_head = _head_mlp(cfg.d_model, 1, "noul_head")
        self.choice_head = _head_mlp(cfg.d_model, 1, "choice_head")
        self.span_head = _head_mlp(cfg.d_model, 2, "span_head")
        self.embed_pool = tf.keras.layers.Dense(cfg.d_model, name="embed_pool")

    @classmethod
    def from_preset(cls, name: str = "arc1-tiny", **overrides) -> "Arc1Model":
        return cls(Arc1Config.from_preset(name, **overrides))

    def select_ladder_depth(self, depth: Optional[int]) -> "Arc1Model":
        """Set the default ladder depth (same weights). Prefer ``depth=`` per call."""
        self.arc1_config = self.arc1_config.with_depth(depth)
        return self

    # ------------------------------------------------------------------ encoder
    def _encode(self, token_ids, training=False, depth=None):
        cfg = self.arc1_config
        active = set(cfg.ladder_block_indices(depth))
        x = self.token_embedding(token_ids)
        x = self.embed_dropout(x, training=training)
        history = [x]
        for i, (residual, block) in enumerate(zip(self.depth_residuals, self.blocks)):
            if i not in active:
                continue
            x = residual(history)
            x = block(x, token_ids=token_ids, training=training)
            history.append(x)
        return self.final_norm(x)

    def _last_hidden(self, hidden, token_ids):
        """Hidden state at the last non-pad position (sees the whole causal prefix)."""
        mask = tf.cast(tf.not_equal(token_ids, self.arc1_config.pad_id), tf.int32)
        last = tf.maximum(tf.reduce_sum(mask, axis=1) - 1, 0)
        return tf.gather(hidden, last, batch_dims=1)

    @staticmethod
    def _masked_mean(hidden, mask):
        mask = tf.cast(mask, hidden.dtype)[..., None]
        total = tf.reduce_sum(hidden * mask, axis=1)
        return total / tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)

    # -------------------------------------------------------------------- heads
    def call(self, token_ids, training=False, depth=None):
        hidden = self._encode(token_ids, training=training, depth=depth)
        return tf.matmul(hidden, self.token_embedding.embeddings, transpose_b=True)

    def decide(self, token_ids, training=False, depth=None, with_lm=False):
        """One encoder pass → raw (uncalibrated) logits for every decision head.

        Returns ``noul`` (B,), ``choice`` (B,), ``span_start``/``span_end`` (B, T),
        and ``hidden`` (B, T, D); plus ``lm`` (B, T, V) when ``with_lm``.
        """
        hidden = self._encode(token_ids, training=training, depth=depth)
        last = self._last_hidden(hidden, token_ids)
        span = self.span_head(hidden)
        out = {
            "hidden": hidden,
            "noul": tf.squeeze(self.noul_head(last), axis=-1),
            "choice": tf.squeeze(self.choice_head(last), axis=-1),
            "span_start": span[..., 0],
            "span_end": span[..., 1],
        }
        if with_lm:
            out["lm"] = tf.matmul(hidden, self.token_embedding.embeddings, transpose_b=True)
        return out

    def pooled_embedding(self, hidden, pool_mask):
        vec = self.embed_pool(self._masked_mean(hidden, pool_mask))
        return tf.nn.l2_normalize(vec, axis=-1)

    def confidence(self, token_ids, training=False, depth=None):
        """Calibrated noul probability read at the last real token."""
        logits = self.decide(token_ids, training=training, depth=depth)["noul"]
        return tf.sigmoid(logits / self.arc1_config.temperature("noul"))

    def embed_text(self, token_ids, training=False, depth=None, pool_mask=None):
        hidden = self._encode(token_ids, training=training, depth=depth)
        if pool_mask is None:
            pool_mask = tf.not_equal(token_ids, self.arc1_config.pad_id)
        return self.pooled_embedding(hidden, pool_mask)

    def build_model(self) -> "Arc1Model":
        dummy = tf.zeros((1, 16), dtype=tf.int32) + 4
        self(dummy, training=False)  # Keras 3 marks the model built only via __call__
        self.decide(dummy, training=False, with_lm=True)
        self.embed_text(dummy, training=False)
        return self

    def compile_model(self, learning_rate=3e-4):
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return self

    def generate(
        self,
        token_ids: Sequence[int],
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_k: int = 40,
        eos_id: Optional[int] = 1,
        allowed_token_ids: Optional[Sequence[int]] = None,
        depth: Optional[int] = None,
    ) -> List[int]:
        cfg = self.arc1_config
        tokens = [int(t) for t in token_ids] or [cfg.pad_id]
        allowed = None if allowed_token_ids is None else [int(i) for i in allowed_token_ids]
        for _ in range(max_new_tokens):
            window = tokens[-cfg.seq_len :]
            logits = self(tf.constant([window], dtype=tf.int32), training=False, depth=depth)[0, -1]
            next_id = int(
                _sample_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    allowed_ids=allowed,
                )
            )
            tokens.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return tokens

    def get_config(self):
        return self.arc1_config.to_dict()

    @classmethod
    def from_config(cls, config):
        return cls(Arc1Config.from_dict(config))

    def get_model_info(self, depth: Optional[int] = None) -> Dict:
        built = int(self.count_params()) if self.built else 0
        return {
            "name": "Arc1Model",
            "built_parameters": built,
            "active_depth": self.arc1_config.resolve_depth(depth),
            "ladder_indices": self.arc1_config.ladder_block_indices(depth),
            "config": self.arc1_config.to_dict(),
        }
