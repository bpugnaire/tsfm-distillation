"""Which layers can be int8? -> results/export/int8_sensitivity_<stem>.json

Quantises the exported fp32 student with different exclusion sets and reports the parity error
error / max(scale, |q - loc|) on the parity windows (p99 and max)."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

from tinycast.eval.benchmark import ROOT
from tinycast.export.onnx import parity_inputs
from tinycast.models.student import StudentForecaster

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stem", required=True)
    a = ap.parse_args()
    m = StudentForecaster.from_run(a.run).model
    fp32 = ROOT / "web" / "public" / "models" / f"{a.stem}.fp32.onnx"
    x = parity_inputs(1000, m.c.context_length)
    with torch.no_grad():
        qt = m(torch.tensor(x)).numpy()
        _, _, loc, sc = m.prepare(torch.tensor(x))
    loc, sc = loc.numpy()[:, :, None], sc.numpy()[:, :, None]
    tmp = Path(tempfile.mkdtemp())
    quant_pre_process(str(fp32), str(tmp / "pre.onnx"), skip_symbolic_shape=True)
    mm = [
        n.name
        for n in onnx.load(str(tmp / "pre.onnx")).graph.node
        if n.op_type in ("MatMul", "Gemm")
    ]
    embed = [n for n in mm if n.startswith("/net/embed/")]
    head = [n for n in mm if n.startswith("/net/head/")]
    variants = {
        "all MatMuls int8": ([], False),
        "keep head fp32": (head, False),
        "keep embedding + head fp32": (embed + head, False),
        "keep embedding + head fp32, per-channel weights": (embed + head, True),
    }
    out = {
        "run": a.run,
        "n_matmuls": len(mm),
        "metric": "abs error / max(scale, |q - loc|)",
        "variants": {},
    }
    for name, (excl, pc) in variants.items():
        q_path = tmp / "q.onnx"
        quantize_dynamic(
            str(tmp / "pre.onnx"),
            str(q_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=excl,
            per_channel=pc,
        )
        q8 = ort.InferenceSession(str(q_path)).run(None, {"context": x})[0]
        r = np.abs(qt - q8) / (sc + np.abs(qt - loc))
        out["variants"][name] = {
            "p99": float(np.quantile(r, 0.99)),
            "max": float(r.max()),
            "size_mb": round(q_path.stat().st_size / 1e6, 3),
        }
        print(
            f"{name:50s} p99 {out['variants'][name]['p99']:.4f}  max {r.max():.3f}  {out['variants'][name]['size_mb']} MB"
        )
    (ROOT / "results" / "export" / f"int8_sensitivity_{a.stem}.json").write_text(
        json.dumps(out, indent=1)
    )
