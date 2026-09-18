"""Causal small language model built from ARCANE layers.

The default ``100m`` preset is a decoder-only LM (~100M trainable weights):
causal Katharopoulos attention, DenseGSER expand, bioplastic projection,
token-parallel resonance, and depth-wise AttentionResidual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from .layers import ArcaneDecoderBlock, PositionalEncodingLayer
from .mechanisms import AttentionResidual, RMSNorm


SLM_PRESETS: Dict[str, Dict] = {
    "tiny": {
        "vocab_size": 512,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_mult": 2,
        "seq_len": 64,
        "dropout_rate": 0.0,
    },
    "100m": {
        "vocab_size": 32000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "ffn_mult": 2,
        "seq_len": 256,
        "dropout_rate": 0.1,
    },
    # Same architecture as "distill", scaled to what a CPU-only box can actually
    # train. ~8.8M parameters at vocab 8000.
    "distill-small": {
        "vocab_size": 8000,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_mult": 4,
        "seq_len": 256,
        "dropout_rate": 0.05,
        "softmax_every": 3,
        "use_rope": True,
        "use_decay": True,
        "reweight_centered": True,
        "gate_normalize": True,
        "ffn_out_activation": None,
        "norm_type": "rms",
        # seq_len-sized chunks: one tf.scan iteration. Identical to chunk_size=64
        # (test_chunk_size_does_not_change_output) and much cheaper on CPU.
        "chunk_size": 256,
    },
    # Tuned as a distillation target for a softmax teacher (Qwen2.5-0.5B).
    # Differences from "100m", each for a specific reason:
    #   softmax_every=4  -> 3 of 12 layers do exact key lookup, which pure
    #                       linear attention cannot express
    #   use_rope         -> same positional geometry as the teacher; also
    #                       replaces the fixed-length sinusoidal buffer
    #   use_decay        -> KV state stops weighting the whole prefix uniformly
    #   ffn_mult=4       -> FFN width is where factual knowledge lands
    #   ffn_out_act=None -> unbiased writes into the residual stream
    #   gate_normalize   -> spike threshold becomes scale-relative and learnable
    #   norm_type=rms    -> matches the teacher, one less distribution mismatch
    "distill": {
        "vocab_size": 32000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "ffn_mult": 4,
        "seq_len": 512,
        "dropout_rate": 0.1,
        "softmax_every": 4,
        "use_rope": True,
        "use_decay": True,
        "reweight_centered": True,
        "gate_normalize": True,
        "ffn_out_activation": None,
        "norm_type": "rms",
    },
}


@dataclass
class ArcaneSLMConfig:
    """Width/depth/vocab for ``ArcaneSmallLanguageModel``."""

    vocab_size: int = 32000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_mult: int = 2
    seq_len: int = 256
    dropout_rate: float = 0.1
    resonance_factor: float = 0.15
    resonance_cycles: int = 3
    spike_threshold: float = 0.4
    leak_rate: float = 0.1
    enable_inference_plasticity: bool = False
    pad_id: int = 0
    # Every ``softmax_every``-th layer uses full softmax attention (0 = all linear).
    softmax_every: int = 0
    use_rope: bool = False
    use_decay: bool = False
    chunk_size: int = 64
    reweight_centered: bool = False
    gate_normalize: bool = False
    ffn_out_activation: Optional[str] = "gelu"
    norm_type: str = "layer"

    def attention_types(self) -> List[str]:
        """Per-layer attention kind, deepest-biased so softmax lands late."""
        if not self.softmax_every:
            return ["linear"] * self.num_layers
        return [
            "softmax" if (i + 1) % self.softmax_every == 0 else "linear"
            for i in range(self.num_layers)
        ]

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ArcaneSLMConfig":
        key = name.lower()
        if key not in SLM_PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {sorted(SLM_PRESETS)}")
        params = dict(SLM_PRESETS[key])
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> Dict:
        return asdict(self)

    def estimate_trainable_parameters(self) -> int:
        """Closed-form count matching the subclassed architecture (tied LM head)."""
        d = self.d_model
        d_ff = d * self.ffn_mult
        norm = d if self.norm_type == "rms" else 2 * d
        embed = self.vocab_size * d
        # QKVO + LN; linear attention adds the semantic reweight head and,
        # optionally, one decay logit per head.
        attn_softmax = 4 * d * d + 4 * d + 2 * d
        attn_linear = attn_softmax + d + 1 + (self.num_heads if self.use_decay else 0)
        gser = 2 * d * d_ff + 2 * d_ff + (d_ff if self.gate_normalize else 0)
        bioplastic = d_ff * d + d
        resonance = d * d + 2 * d + 2 * d
        attnres = 2 * d
        shared = gser + bioplastic + resonance + norm + attnres
        total = embed + norm
        for kind in self.attention_types():
            total += shared + (attn_softmax if kind == "softmax" else attn_linear)
        return total


class ArcaneSmallLanguageModel(tf.keras.Model):
    """Decoder-only ARCANE language model with a tied embedding / LM head."""

    def __init__(self, config: Optional[ArcaneSLMConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.slm_config = config or ArcaneSLMConfig.from_preset("100m")
        cfg = self.slm_config
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.token_embedding = tf.keras.layers.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            name="token_embedding",
        )
        # RoPE is applied inside attention, so the additive sinusoidal table is
        # skipped entirely when it is on -- adding both would double-count position.
        self.positional_encoding = (
            None
            if cfg.use_rope
            else PositionalEncodingLayer(
                max_position=cfg.seq_len,
                d_model=cfg.d_model,
                name="positional_encoding",
            )
        )
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.depth_residuals = [
            AttentionResidual(cfg.d_model, name=f"attnres_{i}")
            for i in range(cfg.num_layers)
        ]
        self.blocks = [
            ArcaneDecoderBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                ffn_mult=cfg.ffn_mult,
                dropout_rate=cfg.dropout_rate,
                resonance_factor=cfg.resonance_factor,
                resonance_cycles=cfg.resonance_cycles,
                spike_threshold=cfg.spike_threshold,
                leak_rate=cfg.leak_rate,
                enable_inference_plasticity=cfg.enable_inference_plasticity,
                attention_type=kind,
                use_rope=cfg.use_rope,
                max_position=max(cfg.seq_len, 2048),
                chunk_size=cfg.chunk_size,
                use_decay=cfg.use_decay,
                reweight_centered=cfg.reweight_centered,
                gate_normalize=cfg.gate_normalize,
                ffn_out_activation=cfg.ffn_out_activation,
                norm_type=cfg.norm_type,
                name=f"decoder_block_{i}",
            )
            for i, kind in enumerate(cfg.attention_types())
        ]
        self.final_norm = (
            RMSNorm(eps=1e-6, name="final_norm")
            if cfg.norm_type == "rms"
            else tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        )

    @classmethod
    def from_preset(cls, name: str = "100m", **overrides) -> "ArcaneSmallLanguageModel":
        return cls(ArcaneSLMConfig.from_preset(name, **overrides))

    def call(self, token_ids, training=False):
        cfg = self.slm_config
        x = self.token_embedding(token_ids)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        x = self.embed_dropout(x, training=training)
        history = [x]
        for residual, block in zip(self.depth_residuals, self.blocks):
            x = residual(history)
            x = block(x, training=training)
            history.append(x)
        x = self.final_norm(x)
        return tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)

    def build_model(self) -> "ArcaneSmallLanguageModel":
        dummy = tf.zeros((1, self.slm_config.seq_len), dtype=tf.int32)
        self(dummy, training=False)
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
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = 1,
        allowed_token_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Autoregressive sampling. Crops the context window to ``seq_len``.

        Short prompts are *not* left-padded. Training never sees pad tokens, and
        sinusoidal positions are ``0..t``, so padding would shift every real token
        to the wrong position and poison causal attention.
        """
        cfg = self.slm_config
        tokens = [int(t) for t in token_ids] or [cfg.pad_id]
        allowed = None if allowed_token_ids is None else [int(i) for i in allowed_token_ids]
        for _ in range(max_new_tokens):
            window = tokens[-cfg.seq_len :]
            logits = self(tf.constant([window], dtype=tf.int32), training=False)[0, -1]
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
        return self.slm_config.to_dict()

    @classmethod
    def from_config(cls, config):
        return cls(ArcaneSLMConfig(**config))

    def get_model_info(self) -> Dict:
        built = 0
        if self.built:
            built = int(self.count_params())
        return {
            "name": "ArcaneSmallLanguageModel",
            "preset_trainable_estimate": self.slm_config.estimate_trainable_parameters(),
            "built_parameters": built,
            "config": self.slm_config.to_dict(),
        }


def _sample_logits(logits, temperature=0.8, top_k=40, allowed_ids=None) -> int:
    logits = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    logits = np.array(logits, dtype=np.float64)
    if allowed_ids is not None:
        allowed = np.asarray(allowed_ids, dtype=np.int32)
        allowed = allowed[(allowed >= 0) & (allowed < logits.shape[-1])]
        if allowed.size:
            masked = np.full_like(logits, -1e9)
            masked[allowed] = logits[allowed]
            logits = masked
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / max(float(temperature), 1e-5)
    if top_k is not None and top_k > 0:
        finite = np.isfinite(logits) & (logits > -1e8)
        k = min(int(top_k), int(np.count_nonzero(finite)) or 1)
        threshold = np.partition(logits, -k)[-k]
        logits = np.where(logits < threshold, -1e9, logits)
    logits = logits - logits.max()
    probs = np.exp(logits)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs = probs / total
    return int(np.random.choice(probs.shape[0], p=probs))


def causal_windows(token_ids: Sequence[int], seq_len: int) -> tuple:
    """Pack a token stream into (x, y) next-token windows of length ``seq_len``."""
    ids = np.asarray(list(token_ids), dtype=np.int32)
    if ids.size < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens, got {ids.size}")
    n = ids.size - seq_len
    x = np.stack([ids[i : i + seq_len] for i in range(n)])
    y = np.stack([ids[i + 1 : i + seq_len + 1] for i in range(n)])
    return x, y


def make_causal_dataset(
    token_ids: Sequence[int],
    seq_len: int,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Non-overlapping (by default strided) causal LM windows as a ``tf.data`` pipeline."""
    ids = np.asarray(list(token_ids), dtype=np.int32)
    step = seq_len if stride is None else int(stride)
    if ids.size < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens, got {ids.size}")
    starts = np.arange(0, ids.size - seq_len, step, dtype=np.int32)
    x = np.stack([ids[s : s + seq_len] for s in starts])
    y = np.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(starts), 10_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
