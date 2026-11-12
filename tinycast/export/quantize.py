"""int8 post-training quantisation of the exported student (onnxruntime.quantization).

Dynamic quantisation: weights of MatMul/Gemm stored as int8 (per output channel), activations
quantised on the fly per batch. No calibration data needed. The scaler ops stay float32.

Sensitivity (debug student, 1000 parity windows, error / max(scale, |q - loc|), p99):
all MatMuls 3.0%; keeping the patch embedding + output head in fp32 and per-channel weights: 1.7%.
The first/last layers touch the raw-scale-sensitive input/output; the head's error is amplified by
the cumulative softplus + sinh. So they are excluded (docs/journal.md, 2026-11-12).
"""

from __future__ import annotations

from pathlib import Path


def quantize_int8(
    fp32_path: Path,
    out_path: Path,
    keep_fp32: tuple[str, ...] = ("/net/embed/", "/net/head/", "/net/inp/"),
) -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

    pre = out_path.with_name(out_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    nodes = onnx.load(str(pre)).graph.node
    exclude = [
        n.name for n in nodes if n.op_type in ("MatMul", "Gemm") and n.name.startswith(keep_fp32)
    ]
    quantize_dynamic(
        str(pre), str(out_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=exclude, per_channel=True,
    )  # fmt: skip
    pre.unlink(missing_ok=True)
    return out_path
