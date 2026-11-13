"""make cache T=<teacher> C=<train config>: run a teacher over the training corpus of a train config.

The cache key is the train config's corpus spec, so a training run and its cache cannot drift apart.
Resumable: re-running skips completed shards.
"""

import argparse

import torch

from tinycast.config import ModelEntry, TrainConfig, load
from tinycast.data.cache import cache_teacher_on_corpus
from tinycast.data.corpus import build_corpus
from tinycast.distill.train import specs
from tinycast.eval.benchmark import ROOT

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
        and entry.kind != "timestral"
    ):
        entry.device = "mps"
    elif entry.device == "auto":
        entry.device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_spec, _ = specs(load(a.train_config, TrainConfig))
    cache_teacher_on_corpus(entry, tr_spec, build_corpus(tr_spec))
