import importlib.util
from pathlib import Path
import sys

from gpbacay_arcane.arc1 import Arc1Config, Arc1LanguageModel


_EXPORT_PATH = Path(__file__).parents[1] / "examples" / "export_arc1_lm.py"
_SPEC = importlib.util.spec_from_file_location("arcane_export_arc1_lm", _EXPORT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exporter
_SPEC.loader.exec_module(exporter)


def test_dynamic_int8_export_runs_and_preserves_logits():
    config = Arc1Config(
        vocab_size=64,
        d_model=32,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        seq_len=32,
        engram_table_size=64,
        engram_rows=2,
        lm_architecture="hybrid",
        lm_block_pattern=("conv", "conv", "attention"),
    )
    model = Arc1LanguageModel(config).build_model()
    payload = exporter.convert_to_tflite(model, seq_len=16, quant="dynamic-int8")
    metrics = exporter.validate_tflite(model, payload, 16, config.vocab_size, runs=1)
    assert payload[4:8] == b"TFL3"
    assert metrics["mean_abs_logit_error"] < 0.15
    # Random, untrained logits contain many near ties, so a one-ULP change can
    # move argmax even when the full-logit error is tiny.  The production CLI
    # retains its stricter 90% gate for trained checkpoints.
    assert metrics["top1_agreement"] >= 0.5
