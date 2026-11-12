"""make cache T=<teacher> C=<train config>: run a teacher over the training corpus of a train config.

The cache key is the train config's corpus spec, so a training run and its cache cannot drift apart.
Resumable: re-running skips completed shards.
"""

import argparse
import importlib.metadata as md

import torch

from tinycast.config import QUANTILE_LEVELS, ModelEntry, TrainConfig, load
from tinycast.data.cache import check_cache, write_cache
from tinycast.data.corpus import build_corpus
from tinycast.distill.train import specs
from tinycast.eval.benchmark import ROOT, git_sha
from tinycast.models import build_model

PKG = {"toto": "toto-2", "t0": "tfc-t0", "chronos2": "chronos-forecasting"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    entry = load(ROOT / "configs" / "teachers" / f"{a.teacher}.yaml", ModelEntry)
    if a.device:
        entry.device = a.device
    elif (
        entry.device == "auto"
        and not torch.cuda.is_available()
        and torch.backends.mps.is_available()
    ):
        entry.device = "mps"
    tr_spec, _ = specs(load(a.train_config, TrainConfig))
    corpus = build_corpus(tr_spec)
    model = build_model(entry)
    meta = {
        "teacher": entry.name,
        "checkpoint": entry.checkpoint,
        "package": f"{PKG[entry.kind]}=={md.version(PKG[entry.kind])}",
        "levels": list(QUANTILE_LEVELS),
        "native_levels": list(model.native_levels),
        "horizon": tr_spec.horizon,
        "context_length": tr_spec.context_length,
        "corpus": vars(tr_spec) | {"key": tr_spec.key()},
        "device": entry.device,
        "git_sha": git_sha(),
    }
    d = write_cache(
        entry.name,
        tr_spec.key(),
        corpus.ids,
        corpus.context,
        lambda c: model.forecast(c, tr_spec.horizon),
        meta,
    )
    print(check_cache(entry.name, tr_spec.key(), len(corpus.ids)), "->", d)
