"""Evaluate a saved checkpoint. Test-split access requires an explicit ``final=True``.

Every test access is appended to ``<run_dir>/test_access.log`` so the number of
times the held-out set was looked at is auditable.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .data import load_npz
from .metrics import binary_metrics, bootstrap_ci
from .models import load_checkpoint
from .train import predict_proba


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
    norm = meta["normalizer"]
    x = torch.from_numpy(((imgs.astype(np.float32) / 255.0 - norm["mean"]) / norm["std"])).unsqueeze(1)
    t0 = time.time()
    p = predict_proba(model, x)
    infer_s = time.time() - t0
    thr_y = meta.get("val_threshold_youden", 0.5)
    res = {
        "checkpoint": ckpt_path, "split": split, "n": int(len(labels)),
        "inference_seconds_total": round(infer_s, 3),
        "at_0.5": binary_metrics(labels, p, 0.5),
        "at_val_youden": binary_metrics(labels, p, thr_y),
        "ci95_at_0.5": bootstrap_ci(labels, p, 0.5, n_boot=n_boot) if n_boot else {},
    }
    if save:
        run_dir = os.path.dirname(ckpt_path)
        np.save(os.path.join(run_dir, f"{split}_scores.npy"), p)
        with open(os.path.join(run_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(res, f, indent=1)
        if split == "test":
            with open(os.path.join(run_dir, "test_access.log"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} test evaluated\n")
    return res
