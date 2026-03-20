"""
Attention Residuals vs Standard Residuals — Proper Test
=========================================================
Correctly implements AttnRes using a Subclassed Model approach so the block
history is maintained as a Python list *during the forward pass* (not baked
into the static graph).  This avoids the graph-explosion / slow-retracing
problem seen in the first attempt.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import time

# ──────────────────────────────────────────────
# 1.  Core AttnRes helper (single call, no state)
# ──────────────────────────────────────────────
class _AttnResOp(layers.Layer):
    """
    Given a list of tensors (batch, d), computes softmax-weighted sum over the
    depth dimension using a single learned pseudo-query wl ∈ R^d.
    """
    def __init__(self, d_model, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.eps = eps

    def build(self, _):
        # pseudo-query: initialised to zero → equal weighting at step 0
        self.wl = self.add_weight(shape=(self.d_model,), initializer="zeros",
                                  trainable=True, name="pseudo_query")
        self.gamma = self.add_weight(shape=(self.d_model,), initializer="ones",
                                     trainable=True, name="rmsnorm_gamma")
        super().build(_)

    def call(self, history):
        # history: list of tensors, each (B, d)
        V = tf.stack(history, axis=1)            # (B, L, d)
        # RMSNorm on keys to prevent magnitude dominance
        rms = tf.sqrt(tf.reduce_mean(V**2, axis=-1, keepdims=True) + self.eps)
        K = (V / rms) * self.gamma               # (B, L, d)
        # scores = wl · k_i  shape: (B, L)
        scores = tf.einsum("d,bld->bl", self.wl, K)
        alpha = tf.nn.softmax(scores, axis=-1)   # (B, L)
        # weighted sum
        out = tf.einsum("bl,bld->bd", alpha, V)  # (B, d)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "eps": self.eps})
        return cfg


# ──────────────────────────────────────────────
# 2.  A single Transformer-like block using AttnRes
# ──────────────────────────────────────────────
class AttnResBlock(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.norm = layers.LayerNormalization()
        self.ff   = layers.Dense(d_model, activation="relu")

    def build(self, _):
        self.attn_res = _AttnResOp(self.d_model, name=self.name + "_op")
        super().build(_)

    def call(self, history):
        # history is the list of all prior block outputs
        x = self.attn_res(history)          # selective depth aggregation
        x = self.norm(x)
        x = self.ff(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


# ──────────────────────────────────────────────
# 3.  Full models (subclassed, forward pass manages history)
# ──────────────────────────────────────────────
class StandardResidualNet(Model):
    def __init__(self, d_model, num_layers, num_classes):
        super().__init__()
        self.embed   = layers.Dense(d_model, activation="relu")
        self.flatten = layers.Flatten()
        self.blocks  = [
            tf.keras.Sequential([
                layers.LayerNormalization(),
                layers.Dense(d_model, activation="relu")
            ], name=f"std_block_{i}")
            for i in range(num_layers)
        ]
        self.head = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.embed(x)
        for block in self.blocks:
            residual = x
            x = block(x, training=training) + residual
        return self.head(x)


class AttentionResidualNet(Model):
    def __init__(self, d_model, num_layers, num_classes):
        super().__init__()
        self.embed   = layers.Dense(d_model, activation="relu")
        self.flatten = layers.Flatten()
        self.blocks  = [AttnResBlock(d_model, name=f"attn_res_block_{i}")
                        for i in range(num_layers)]
        self.head    = layers.Dense(num_classes, activation="softmax")
        # Final AttnRes op over all outputs before classifier
        self._final_op = _AttnResOp(d_model, name="final_attn_res_op")

    def call(self, x, training=False):
        x = self.flatten(x)
        h0 = self.embed(x)
        history = [h0]
        for block in self.blocks:
            h = block(history)
            history.append(h)
        # Final selective read from all depth sources
        out = self._final_op(history)
        return self.head(out)


# ──────────────────────────────────────────────
# 4.  Experiment
# ──────────────────────────────────────────────
def run_experiment():
    print("=" * 60)
    print("  Attention Residuals vs Standard Residuals — Test")
    print("=" * 60)

    # --- Data ---
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:12000].astype("float32") / 255.0
    y_train = y_train[:12000]
    x_test  = x_test[:2000].astype("float32") / 255.0
    y_test  = y_test[:2000]

    D_MODEL     = 128
    NUM_LAYERS  = 8       # moderate depth to show AttnRes benefit
    EPOCHS      = 12
    BATCH_SIZE  = 64
    NUM_CLASSES = 10

    results = {}

    for name, ModelClass in [("Standard Residuals", StandardResidualNet),
                              ("Attention Residuals", AttentionResidualNet)]:

        model = ModelClass(D_MODEL, NUM_LAYERS, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print(f"\n{'─'*60}")
        print(f"  Training: {name}")
        print(f"{'─'*60}")
        t0 = time.time()
        hist = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        elapsed = time.time() - t0

        val_accs = hist.history["val_accuracy"]
        val_losses = hist.history["val_loss"]
        results[name] = {
            "final_val_acc":  val_accs[-1],
            "best_val_acc":   max(val_accs),
            "final_val_loss": val_losses[-1],
            "best_val_loss":  min(val_losses),
            "time_s":         elapsed,
            "val_accs":       val_accs,
        }

    # ──────────────────────────────────────────
    # 5.  Print results summary
    # ──────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<28} {'Standard':>14} {'AttnRes':>14}"
    print(header)
    print("-" * 60)

    std  = results["Standard Residuals"]
    attn = results["Attention Residuals"]

    metrics = [
        ("Final Val Accuracy",    std["final_val_acc"],   attn["final_val_acc"]),
        ("Best Val Accuracy",     std["best_val_acc"],    attn["best_val_acc"]),
        ("Final Val Loss",        std["final_val_loss"],  attn["final_val_loss"]),
        ("Best Val Loss",         std["best_val_loss"],   attn["best_val_loss"]),
        ("Training Time (s)",     std["time_s"],          attn["time_s"]),
    ]
    for label, sv, av in metrics:
        print(f"  {label:<26} {sv:>14.4f} {av:>14.4f}")

    print("-" * 60)
    delta_acc  = attn["final_val_acc"]  - std["final_val_acc"]
    delta_best = attn["best_val_acc"]   - std["best_val_acc"]
    delta_loss = attn["final_val_loss"] - std["final_val_loss"]
    print(f"  {'ΔVal Acc (Final)':<26} {delta_acc:>+14.4f}")
    print(f"  {'ΔVal Acc (Best)':<26}  {delta_best:>+13.4f}")
    print(f"  {'ΔVal Loss (Final)':<26} {delta_loss:>+14.4f}")
    print("=" * 60)

    # Per-epoch val accuracy trace
    print("\n  Epoch-by-Epoch Val Accuracy:")
    print(f"  {'Epoch':<8} {'Standard':>12} {'AttnRes':>12} {'Δ':>8}")
    print("  " + "-" * 44)
    for ep, (sa, aa) in enumerate(zip(std["val_accs"], attn["val_accs"]), 1):
        print(f"  {ep:<8} {sa:>12.4f} {aa:>12.4f} {aa-sa:>+8.4f}")

    print("\n  Verdict:")
    if attn["best_val_acc"] > std["best_val_acc"]:
        print(f"  ✅ Attention Residuals IMPROVED best accuracy by {delta_best*100:+.2f}%")
    else:
        print(f"  ⚠️  Attention Residuals did not beat Standard Residuals in best acc.")
    if attn["final_val_acc"] > std["final_val_acc"]:
        print(f"  ✅ Attention Residuals IMPROVED final accuracy by {delta_acc*100:+.2f}%")
    else:
        print(f"  ℹ️  Final accuracy: AttnRes ({attn['final_val_acc']:.4f}) vs Std ({std['final_val_acc']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    run_experiment()
