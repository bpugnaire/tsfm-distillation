"""Export the student to ONNX *including the scaler*, and check parity with PyTorch.

Graph contract (what the browser relies on):
    input  "context":   float32 (batch, 512), raw values, NaN = missing, left-pad short series with NaN
    output "quantiles": float32 (batch, 13, 64), raw-space quantiles at QUANTILE_LEVELS
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tinycast.config import QUANTILE_LEVELS
from tinycast.models.base import check_output
from tinycast.models.student import Student


def export_onnx(model: Student, path: Path, opset: int = 17) -> Path:
    model = model.eval().cpu()
    T = model.c.context_length
    dummy = torch.full((2, T), 10.0) + torch.randn(2, T)
    path.parent.mkdir(parents=True, exist_ok=True)
    # nn.TransformerEncoder's fused "fast path" kernel has no ONNX symbolic: disable it.
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["context"],
            output_names=["quantiles"],
            dynamic_axes={"context": {0: "batch"}, "quantiles": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    return path


def parity_inputs(n: int, T: int, seed: int = 0) -> np.ndarray:
    """Windows that exercise the scaler: several scales/offsets, NaN gaps, left padding."""
    from tinycast.data.synthetic import synthetic_batch

    rng = np.random.default_rng(seed)
    ctx, _ = synthetic_batch(np.arange(n) + 500_000, T, 64)
    ctx = ctx * rng.choice([1e-3, 1.0, 1e3, 5e4], size=(n, 1)) + rng.choice(
        [0.0, -50.0, 1e4], size=(n, 1)
    )
    for i in range(0, n, 7):  # gaps
        s = rng.integers(0, T - 40)
        ctx[i, s : s + rng.integers(5, 40)] = np.nan
    for i in range(3, n, 11):  # short series, left-padded
        ctx[i, : rng.integers(50, 400)] = np.nan
    return ctx.astype(np.float32)


def parity_report(model: Student, onnx_path: Path, n: int = 1000, seed: int = 0) -> dict:
    import onnxruntime as ort

    x = parity_inputs(n, model.c.context_length, seed)
    with torch.no_grad():
        q_t = model.eval().cpu()(torch.tensor(x)).numpy()
        _, _, loc, scale = model.prepare(torch.tensor(x))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    q_o = np.concatenate([sess.run(None, {"context": x[i : i + 128]})[0] for i in range(0, n, 128)])
    # float32 rounding grows with magnitude (sinh amplifies it in deep tails), so the tolerance is
    # relative to max(series scale, |q - loc|), not to the scale alone.
    sc, lc = scale.numpy()[:, :, None], loc.numpy()[:, :, None]
    rel = np.abs(q_t - q_o) / (sc + np.abs(q_t - lc))
    return {
        "n": n,
        "max_abs_diff": float(np.abs(q_t - q_o).max()),
        "max_rel_diff": float(rel.max()),
        "p99_rel_diff": float(np.quantile(rel, 0.99)),
        "finite": bool(np.isfinite(q_o).all()),
        "monotone": bool((np.diff(q_o, axis=1) >= -1e-5 * (1 + np.abs(q_o[:, 1:]))).all()),
        "size_mb": round(onnx_path.stat().st_size / 1e6, 3),
    }


class OnnxForecaster:
    """Forecaster protocol on top of an exported ONNX file (fp32 or int8)."""

    quantile_levels = QUANTILE_LEVELS
    native_levels = QUANTILE_LEVELS

    def __init__(self, path: str, name: str = "student-onnx", threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name = name
        meta = Path(path).with_suffix(".json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        self.n_params = info.get("n_params", 0)
        self.context_length = self.sess.get_inputs()[0].shape[1]

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        x = np.asarray(context, dtype=np.float32)
        T = self.context_length
        if x.shape[1] < T:
            x = np.pad(x, ((0, 0), (T - x.shape[1], 0)), constant_values=np.nan)
        q = self.sess.run(None, {"context": x[:, -T:]})[0][..., :horizon]
        return check_output(q, len(x), horizon, self.name)
