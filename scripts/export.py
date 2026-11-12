"""make export R=runs/<run>: ONNX fp32 + int8 + parity -> web/public/models/ + results/export/<run>.json"""

import argparse
import json
from pathlib import Path

from tinycast.eval.benchmark import ROOT
from tinycast.export.onnx import export_onnx, parity_report
from tinycast.export.quantize import quantize_int8
from tinycast.models.student import StudentForecaster

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--name", default=None, help="file stem in web/public/models (default: run name)"
    )
    a = ap.parse_args()
    run = Path(a.run)
    sf = StudentForecaster.from_run(str(run))
    stem = a.name or run.name.split("_", 1)[1]
    out = ROOT / "web" / "public" / "models"
    fp32 = export_onnx(sf.model, out / f"{stem}.fp32.onnx")
    int8 = quantize_int8(fp32, out / f"{stem}.int8.onnx")
    info = {
        "run": run.name,
        "n_params": sf.n_params,
        "student_config": json.loads((run / "student_config.json").read_text()),
    }
    for p in (fp32, int8):
        p.with_suffix(".json").write_text(json.dumps(info))
    rep = {
        "run": run.name,
        "n_params": sf.n_params,
        "fp32": parity_report(sf.model, fp32),
        "int8": parity_report(sf.model, int8),
    }
    (ROOT / "results" / "export").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "export" / f"{stem}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["fp32"]["max_rel_diff"] > 1e-4:
        raise SystemExit("fp32 parity FAILED")
