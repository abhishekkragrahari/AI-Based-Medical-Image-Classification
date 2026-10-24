"""Evaluate a saved checkpoint. Test-split access requires an explicit ``final=True``.

Every test access is appended to ``<run_dir>/test_access.log`` so the number of
times the held-out set was looked at is auditable.

Binary checkpoints report metrics at 0.5 and at the val-chosen thresholds; multi-class
checkpoints report argmax metrics. Saved ``{split}_scores.npy`` is P(class 1) for binary
and the N x K probability matrix for multi-class.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import CLASS_NAMES
from .data import load_npz, to_model_input
from .metrics import binary_metrics, bootstrap_ci, bootstrap_ci_multiclass, multiclass_metrics
from .models import load_checkpoint
from .train import predict_proba_all


def evaluate_checkpoint(ckpt_path: str, data_path: str, split: str = "val", final: bool = False,
                        n_boot: int = 1000, save: bool = True) -> dict:
    if split == "test" and not final:
        raise PermissionError("Test evaluation is reserved for the final run: pass final=True / --final.")
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    model, meta = load_checkpoint(ckpt_path)
    imgs, labels = load_npz(data_path)[split]
    if imgs.shape[1] != meta["input_size"]:
        raise ValueError(f"checkpoint expects {meta['input_size']} px input, data is {imgs.shape[1]} px")
    x = to_model_input(imgs, meta["normalizer"])
    if x.shape[1] != meta.get("in_channels", 1):
        raise ValueError(f"checkpoint expects {meta.get('in_channels', 1)} channel(s), data has {x.shape[1]}")
    names = meta.get("class_names", list(CLASS_NAMES))
    t0 = time.time()
    proba = predict_proba_all(model, x)
    infer_s = time.time() - t0
    res = {"checkpoint": ckpt_path, "split": split, "n": int(len(labels)),
           "inference_seconds_total": round(infer_s, 3)}
    if proba.shape[1] == 2:
        p = proba[:, 1]
        thr_y = meta.get("val_threshold_youden", 0.5)
        res.update({
            "at_0.5": binary_metrics(labels, p, 0.5, tuple(names)),
            "at_val_youden": binary_metrics(labels, p, thr_y, tuple(names)),
            **({"at_val_youden_dedup": binary_metrics(labels, p, meta["val_threshold_youden_dedup"],
                                                      tuple(names))}
               if "val_threshold_youden_dedup" in meta else {}),
            "ci95_at_0.5": bootstrap_ci(labels, p, 0.5, n_boot=n_boot) if n_boot else {},
        })
        scores = p
    else:
        res.update({
            "at_argmax": multiclass_metrics(labels, proba, names),
            "ci95_at_argmax": bootstrap_ci_multiclass(labels, proba, n_boot=n_boot) if n_boot else {},
        })
        scores = proba
    if save:
        run_dir = os.path.dirname(ckpt_path)
        np.save(os.path.join(run_dir, f"{split}_scores.npy"), scores)
        with open(os.path.join(run_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(res, f, indent=1)
        if split == "test":
            with open(os.path.join(run_dir, "test_access.log"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} test evaluated\n")
    return res
