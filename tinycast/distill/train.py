"""Plain PyTorch training loop for the student (GT-only or distillation).

runs/<date>_<name>/ gets: config.yaml, student_config.json, metrics.json, best.pt, log.txt.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tinycast.config import CORE_LEVELS, TrainConfig, to_dict
from tinycast.data.corpus import CorpusSpec, build_corpus
from tinycast.distill import losses as L
from tinycast.eval import metrics as M
from tinycast.models.student import Student

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed the legacy global RNG too, for any library using it
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Linear warm-up then cosine decay to 10% of base."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def specs(cfg: TrainConfig) -> tuple[CorpusSpec, CorpusSpec]:
    d, s = cfg.data, cfg.student
    if d.mix:
        common = dict(
            seed=d.seed,
            context_length=s.context_length,
            horizon=s.horizon,
            rollout_max_blocks=d.rollout_max_blocks,
        )
        tr = CorpusSpec(
            "train", mix=tuple(sorted(d.mix.items())), var_context=d.var_context, **common
        )
        va = CorpusSpec("val", mix=tuple(sorted((d.val_mix or d.mix).items())), **common)
        return tr, va
    n_syn = int(round(d.n_train_windows * d.synthetic_share))
    n_syn_val = int(round(d.n_val_windows * d.synthetic_share))
    tr = CorpusSpec("train", d.n_train_windows - n_syn, n_syn, d.seed, s.context_length, s.horizon)
    va = CorpusSpec(
        "val", d.n_val_windows - n_syn_val, n_syn_val, d.seed, s.context_length, s.horizon
    )
    return tr, va


def scaled_targets(
    model: Student, y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Ground truth / teacher values in the student's scaled space (context statistics only)."""
    y0 = torch.nan_to_num(y, nan=0.0)
    return model.scaler.scale(
        y0, loc if y.dim() == 2 else loc[..., None], scale if y.dim() == 2 else scale[..., None]
    )


@torch.no_grad()
def evaluate(model: Student, ctx: np.ndarray, fut: np.ndarray, device, bs: int = 512) -> dict:
    model.eval()
    qs = []
    for i in range(0, len(ctx), bs):
        x = torch.as_tensor(ctx[i : i + bs], device=device)
        qs.append(model(x).float().cpu().numpy())
    model.train()
    q = np.concatenate(qs)
    levels = model.c.quantile_levels
    w = M.wql(fut, q, levels)
    return {
        "val_wql_geomean": M.geometric_mean(w),
        "val_wql_median": float(np.nanmedian(w)),
        "val_cov80": float(np.nanmean(M.coverage(fut, q, levels, 0.1, 0.9))),
    }


def train(
    cfg: TrainConfig,
    out_root: Path | None = None,
    log_every: int = 50,
    corpora: tuple | None = None,
    teacher_corpus_key: str | None = None,
    init_from: Path | None = None,
    extra_metrics: dict | None = None,
) -> Path:
    """Train a student. `corpora` = (train, val) Corpus objects to use instead of building them from the config
    (fine-tuning on a target domain); `init_from` = a run dir whose best.pt initialises the weights."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    stamp = dt.datetime.now().strftime("%Y-%m-%d")
    run = (out_root or ROOT / "runs") / f"{stamp}_{cfg.name}"
    k = 1
    while run.exists():  # never overwrite a previous run
        k += 1
        run = (out_root or ROOT / "runs") / f"{stamp}_{cfg.name}_{k}"
    run.mkdir(parents=True)
    logf = open(run / "log.txt", "a")  # noqa: SIM115 - kept open for the whole run, closed at the end

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from tinycast.eval.benchmark import git_sha

    (run / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    (run / "student_config.json").write_text(json.dumps(to_dict(cfg.student)))

    tr_spec, va_spec = specs(cfg)
    t0 = time.time()
    tr, va = corpora if corpora is not None else (build_corpus(tr_spec), build_corpus(va_spec))
    src = "custom" if corpora is not None else f"key {tr_spec.key()}"
    log(f"corpus train={len(tr.ids)} val={len(va.ids)} ({src}) in {time.time() - t0:.0f}s")

    levels = cfg.student.quantile_levels
    teacher_q = None
    teacher_diag: dict = {}
    if cfg.loss.alpha > 0:
        from tinycast.data.cache import read_cache

        if not cfg.data.teacher:
            raise ValueError("alpha > 0 requires data.teacher")
        from tinycast.distill.weighting import combine, parse

        kind, members, tau = parse(cfg.data.teacher)
        mans, qs = [], []
        if (
            kind == "ensemble"
        ):  # averaged on disk, shard by shard: one cache in RAM instead of len(members) + a stack
            from tinycast.data.cache import build_ensemble_cache, ensemble_cache_name

            build_ensemble_cache(members, teacher_corpus_key or tr_spec.key(), log=log)
            members, kind = [ensemble_cache_name(members)], "single"
        big = isinstance(tr.context, np.memmap)  # labels of a memory-mapped corpus stay on disk too
        if big and kind != "single":
            raise ValueError("weighted teachers on a memory-mapped corpus are not supported")
        for name in members:
            if big:
                from tinycast.data.cache import read_cache_mmap

                q_m, man = read_cache_mmap(name, teacher_corpus_key or tr_spec.key(), tr.ids)
            else:
                q_m, man = read_cache(name, teacher_corpus_key or tr_spec.key(), tr.ids)
            qs.append(q_m)
            mans.append(man)
        t_levels = tuple(mans[0]["levels"])
        if any(tuple(m["levels"]) != t_levels for m in mans):
            raise ValueError("teachers were cached on different level grids")
        if big:  # single teacher: no stacking copy
            teacher_q, diag = qs[0], {}
        else:
            teacher_q, diag = combine(kind, qs, tr.future, t_levels, tau)
        native = set.intersection(*(set(m["native_levels"]) for m in mans))
        if diag:
            log(f"teacher combination {kind}: {diag}")
            teacher_diag = diag
        t_mask = torch.tensor([lv in native for lv in t_levels], device=device)
        if cfg.loss.kd_levels == "all":
            t_mask = torch.ones_like(t_mask)
        level_idx = [t_levels.index(lv) for lv in levels]
        log(f"teacher {cfg.data.teacher}: native levels {sorted(native)}")

    model = Student(cfg.student)
    if init_from is not None:
        model.load_state_dict(
            torch.load(Path(init_from) / "best.pt", map_location="cpu", weights_only=True)
        )
        log(f"initialised from {init_from}")
    model = model.to(device)
    log(f"student {cfg.student.arch}: {model.n_params() / 1e6:.3f}M params on {device}")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.optim.lr,
    )
    use_amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hw = L.horizon_weights(cfg.student.horizon, cfg.loss.horizon_weighting, device)

    mmapped = isinstance(
        tr.context, np.memmap
    )  # large corpora stay on disk; batches are gathered from the file
    ctx_all = None if mmapped else torch.as_tensor(tr.context)
    fut_all = torch.as_tensor(tr.future)
    if (
        teacher_q is not None and cfg.data.gt_sources
    ):  # hybrid target: real future for these sources
        if isinstance(teacher_q, np.memmap):
            raise ValueError("gt_sources is not supported with memory-mapped labels yet")
        rows = np.isin(tr.source, cfg.data.gt_sources) & np.isfinite(tr.future).all(axis=1)
        teacher_q = np.array(teacher_q, dtype=np.float32, copy=False)
        teacher_q[rows] = np.asarray(tr.future, dtype=np.float32)[rows][:, None, :]
        log(
            f"gt_sources {cfg.data.gt_sources}: {int(rows.sum())} windows trained on the real future"
        )
    tq_mm = isinstance(teacher_q, np.memmap)
    tq_all = torch.as_tensor(teacher_q) if teacher_q is not None and not tq_mm else None
    rng = np.random.default_rng(cfg.seed)
    o = cfg.optim
    history, best, bad_evals = [], float("inf"), 0
    t_train = time.time()
    step = 0
    for step in range(o.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, o.lr, o.warmup_steps, o.max_steps)
        opt.zero_grad(set_to_none=True)
        for _ in range(o.grad_accum):
            raw = rng.integers(0, len(tr.ids), o.batch_size)
            idx = torch.as_tensor(
                np.sort(raw) if mmapped else raw
            )  # sorted: sequential-ish reads from the file
            x = (
                torch.from_numpy(np.asarray(tr.context[idx.numpy()])) if mmapped else ctx_all[idx]
            ).to(device)
            y = fut_all[idx].to(device)
            if (
                cfg.data.random_truncate > 0
            ):  # random history length (see DataConfig.random_truncate)
                T = x.shape[1]
                g = torch.Generator().manual_seed(int(rng.integers(2**31)))
                pick = torch.rand(len(x), generator=g) < cfg.data.random_truncate
                u = torch.rand(len(x), generator=g)
                keep = torch.exp(math.log(64) + u * (math.log(T) - math.log(64))).round()
                keep = torch.where(pick, keep, torch.full_like(keep, T)).to(device)
                cut = (T - keep)[:, None]
                x = torch.where(
                    torch.arange(T, device=device)[None] < cut, torch.full_like(x, float("nan")), x
                )
            with torch.autocast("cuda", enabled=use_amp):
                zq, loc, scale = model.forward_scaled(x)
            zq = zq.float()
            ys = scaled_targets(model, y, loc, scale)
            l_gt = L.pinball_gt(zq, ys, levels, mask=~torch.isnan(y), horizon_w=hw)
            l_kd = None
            if tq_all is not None or tq_mm:
                tq = torch.from_numpy(np.asarray(teacher_q[idx.numpy()])) if tq_mm else tq_all[idx]
                qt = scaled_targets(model, tq.to(device), loc, scale)
                if cfg.loss.kd == "pinball_teacher":
                    l_kd = L.kd_pinball_teacher(zq, qt, levels, t_levels, hw, t_mask)
                elif cfg.loss.kd == "quantile_huber":
                    lm = t_mask[level_idx]
                    l_kd = L.kd_quantile_huber(zq, qt, level_idx, horizon_w=hw, level_mask=lm)
                else:
                    raise ValueError(cfg.loss.kd)
            loss = L.mixed_objective(l_gt, l_kd, cfg.loss.alpha) / o.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
        scaler.step(opt)
        scaler.update()
        if step % log_every == 0:
            kd = f" kd {l_kd.item():.4f}" if l_kd is not None else ""
            log(
                f"step {step:5d} loss {loss.item() * o.grad_accum:.4f} gt {l_gt.item():.4f}{kd} gnorm {gnorm.item():.2f}"
            )
        if (step + 1) % o.eval_every == 0 or step + 1 == o.max_steps:
            ev = evaluate(model, va.context, va.future, device)
            ev.update(step=step + 1, elapsed_s=round(time.time() - t_train, 1))
            history.append(ev)
            improved = ev["val_wql_geomean"] < best
            if improved:
                best, bad_evals = ev["val_wql_geomean"], 0
                torch.save(model.state_dict(), run / "best.pt")
            else:
                bad_evals += 1
            log(
                f"eval step {step + 1}: val WQL(geomean) {ev['val_wql_geomean']:.4f} cov80 {ev['val_cov80']:.3f}{' *' if improved else ''}"
            )
            if bad_evals >= o.patience:
                log(f"early stopping after {bad_evals} evals without improvement")
                break
    metrics = {
        "name": cfg.name,
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "n_params": model.n_params(),
        "device": str(device),
        "train_windows": len(tr.ids),
        "val_windows": len(va.ids),
        "corpus_keys": {"train": tr_spec.key(), "val": va_spec.key()}
        if corpora is None
        else "custom",
        "init_from": str(init_from) if init_from else None,
        "teacher": cfg.data.teacher,
        "teacher_combination": teacher_diag,
        **(extra_metrics or {}),
        "steps": step + 1,
        "wall_seconds": round(time.time() - t_train, 1),
        "best_val_wql_geomean": best,
        "history": history,
        "eval_levels": list(CORE_LEVELS),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"done: best val WQL {best:.4f} -> {run}")
    logf.close()
    return run
