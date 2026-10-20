"""Training loop with validation-only checkpoint selection.

The test split is never touched here; see ``medxai.evaluate`` for the final,
explicitly-gated test evaluation.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from . import __version__
from .data import ArrayDataset, Normalizer, augment_batch, load_npz, md5sum, train_duplicate_mask
from .faithful import cfr_penalty
from .metrics import binary_metrics, youden_threshold
from .models import build_model, count_parameters, save_checkpoint


@dataclass
class TrainConfig:
    data_path: str = "data/pneumoniamnist.npz"
    model: str = "small_cnn"
    model_kwargs: dict = field(default_factory=dict)
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # adamw | adam
    scheduler: str = "cosine"       # cosine | multistep
    milestones: list = field(default_factory=lambda: [50, 75])  # multistep only
    patience: int = 8               # early stopping on val AUROC
    class_weighting: bool = False   # inverse-frequency CE weights from the training split
    augment: bool = True
    seed: int = 0
    num_threads: int = 0            # 0 = torch default
    out_dir: str = "runs"
    run_name: str | None = None
    max_train_samples: int | None = None  # for smoke tests only
    # E9 CAM-faithfulness regularisation (medxai/faithful.py); 0 = off (plain cross-entropy)
    cam_del_weight: float = 0.0
    cam_eq_weight: float = 0.0
    cam_del_frac: float = 0.2

    @classmethod
    def from_file(cls, path: str, **overrides) -> "TrainConfig":
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "medxai", "configs"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {
        "medxai": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "git_commit": commit, "git_dirty": dirty,
    }


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(torch.softmax(model(x[i : i + batch_size]), dim=1)[:, 1])
    return torch.cat(out).numpy()


def train(cfg: TrainConfig, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    data = load_npz(cfg.data_path)
    tr_x, tr_y = data["train"]
    if cfg.max_train_samples:
        tr_x, tr_y = tr_x[: cfg.max_train_samples], tr_y[: cfg.max_train_samples]
    va_x, va_y = data["val"]
    del data  # the test split is deliberately dropped here

    norm = Normalizer.fit(tr_x)
    train_ds, val_ds = ArrayDataset(tr_x, tr_y, norm), ArrayDataset(va_x, va_y, norm)

    model = build_model(cfg.model, **cfg.model_kwargs)
    use_cfr = bool(cfg.cam_del_weight or cfg.cam_eq_weight)
    if use_cfr and not hasattr(model, "cam"):
        raise ValueError("CAM-faithfulness regularisation needs a model with features()/cam() (small_cnn)")
    weight = None
    if cfg.class_weighting:
        counts = np.bincount(tr_y.astype(int), minlength=2).astype(float)
        weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"unknown optimizer '{cfg.optimizer}'")
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.scheduler == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=0.1)
    else:
        raise ValueError(f"unknown scheduler '{cfg.scheduler}'")

    run_name = cfg.run_name or f"{cfg.model}_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(cfg.seed)

    best = {"auroc": -1.0, "epoch": -1}
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_ds), generator=gen)
        total, nb, aux = 0.0, 0, {}
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = train_ds.x[idx], train_ds.y[idx]
            x_clean, theta = xb, None
            if cfg.augment:
                xb, theta = augment_batch(xb, gen, return_theta=True)
            opt.zero_grad()
            if use_cfr:
                feats = model.features(xb)
                loss = loss_fn(model.head(feats), yb)
                pen, parts = cfr_penalty(model, xb, feats, yb, x_clean, theta, cfg.cam_del_weight,
                                         cfg.cam_eq_weight, cfg.cam_del_frac)
                loss = loss + pen
                for k, v in parts.items():
                    aux[k] = aux.get(k, 0.0) + v
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        val_p = predict_proba(model, val_ds.x)
        vm = binary_metrics(va_y, val_p)
        rec = {"epoch": epoch, "train_loss": total / max(nb, 1), "val_auroc": vm["auroc"],
               "val_bal_acc": vm["balanced_accuracy"], "val_acc": vm["accuracy"],
               "elapsed_s": round(time.time() - t0, 1)}
        rec.update({f"train_{k}_loss": v / max(nb, 1) for k, v in aux.items()})
        history.append(rec)
        if verbose:
            print(json.dumps(rec))
        if vm["auroc"] > best["auroc"]:
            best = {"auroc": vm["auroc"], "epoch": epoch}
            meta = {"model": cfg.model, "model_kwargs": cfg.model_kwargs, "normalizer": norm.to_dict(),
                    "input_size": int(tr_x.shape[1]), "epoch": epoch, "config": asdict(cfg),
                    "medxai_version": __version__}
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, meta)
        elif epoch - best["epoch"] >= cfg.patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    # Reload best checkpoint, compute validation predictions and a val-chosen threshold.
    from .models import load_checkpoint

    model, meta = load_checkpoint(os.path.join(run_dir, "best.pt"))
    val_p = predict_proba(model, val_ds.x)
    thr = youden_threshold(va_y, val_p)
    meta["val_threshold_youden"] = thr
    # The official val split shares exact copies with train; a de-duplicated view gives a
    # less optimistic validation estimate and an alternative threshold (both val-only).
    keep = ~train_duplicate_mask(tr_x, va_x)
    dedup = None
    if len(np.unique(va_y[keep])) == 2:
        thr_d = youden_threshold(va_y[keep], val_p[keep])
        meta["val_threshold_youden_dedup"] = thr_d
        dedup = {"n": int(keep.sum()), "n_removed": int((~keep).sum()), "threshold_youden": thr_d,
                 "at_0.5": binary_metrics(va_y[keep], val_p[keep], 0.5)}
    save_checkpoint(os.path.join(run_dir, "best.pt"), model, meta)
    summary = {
        "run_dir": run_dir, "best_epoch": best["epoch"], "train_time_s": round(time.time() - t0, 1),
        "n_params": count_parameters(model), "n_train": int(len(tr_y)), "n_val": int(len(va_y)),
        "val_at_0.5": binary_metrics(va_y, val_p, 0.5),
        "val_at_youden": binary_metrics(va_y, val_p, thr),
        "val_dedup": dedup,
        "data_md5": md5sum(cfg.data_path), "config": asdict(cfg), "env": environment_info(),
    }
    np.save(os.path.join(run_dir, "val_scores.npy"), val_p)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
