"""Phase 1.5: teacher throughput (windows/s) at context 512 / horizon 64 -> results/throughput.json"""

import argparse
import json
import time

import numpy as np
import torch

from tinycast.config import ModelEntry
from tinycast.eval.benchmark import ROOT
from tinycast.eval.latency import hardware
from tinycast.models import build_model

TEACHERS = {
    "toto-2.0-22m": ("toto", "Datadog/Toto-2.0-22m"),
    "toto-2.0-313m": ("toto", "Datadog/Toto-2.0-313m"),
    "t0-gamma": ("t0", "theforecastingcompany/t0-gamma"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    out_path = ROOT / "results" / "throughput.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    hw = hardware()
    rng = np.random.default_rng(0)
    x = (
        10 + np.sin(np.arange(512) * 2 * np.pi / 24)[None] + rng.normal(0, 0.2, (a.batch, 512))
    ).astype(np.float32)
    for dev in a.devices.split(","):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        for name, (kind, ck) in TEACHERS.items():
            try:
                m = build_model(
                    ModelEntry(kind=kind, name=name, checkpoint=ck, device=dev, batch_size=a.batch)
                )
                m.forecast(x[:8], 64)
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < a.seconds:
                    m.forecast(x, 64)
                    n += len(x)
                wps = n / (time.perf_counter() - t0)
                r = {
                    "windows_per_s": round(wps, 1),
                    "batch": a.batch,
                    "device": dev,
                    "hardware": hw,
                }
            except Exception as e:  # record, never hide
                r = {"error": f"{type(e).__name__}: {e}"[:300], "device": dev}
            res[f"{name}@{dev}:{hw['cpu']}"] = r
            print(name, dev, r.get("windows_per_s", r.get("error")), flush=True)
            del m
    out_path.write_text(json.dumps(res, indent=1))
