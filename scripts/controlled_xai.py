"""Revision R1: controlled saliency-faithfulness comparison (architecture x resolution).

Runs every seed of each configuration on the same validation images, so architecture
and resolution can be separated. For each image the deletion gap
(random-order deletion AUC - method deletion AUC; > 0 = more faithful than random)
is averaged over seeds. Paired bootstrap CIs are computed over images for:
  * each method's gap from random, in each configuration;
  * 64 px - 28 px (same architecture) and ResNet-18 - small CNN (same resolution).
Occlusion uses resolution-scaled defaults (patch = side/7, stride = side/14).
Two fills are scored: zero (training mean) and blur.

Usage:
  python -m scripts.controlled_xai \
     --config cnn28=runs/gpu/s28/small_cnn_s{0,1,2}/best.pt@data/pneumoniamnist.npz ... \
     --out reports/revision/controlled_xai.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from medxai.data import load_npz, to_model_input
from medxai.models import load_checkpoint
from medxai.train import resolve_device
from medxai.xai import deletion_auc, explain, random_deletion_auc

METHODS = ("gradcam", "ig", "occlusion")


def boot_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))), "n": int(len(v))}


def feature_map_size(model, x):
    out = {}
    h = model.gradcam_layer.register_forward_hook(lambda m, i, o: out.__setitem__("s", tuple(o.shape[-2:])))
    with torch.no_grad():
        model(x)
    h.remove()
    return list(out["s"])


def run_config(ckpts, data, idx, fills, dev):
    imgs, y = load_npz(data)["val"]
    res = {"ckpts": ckpts, "data": data, "per_seed": []}
    for c in ckpts:
        model, meta = load_checkpoint(c)
        model = model.to(dev).eval()
        x = to_model_input(imgs, meta["normalizer"]).to(dev)
        res["feature_map"] = feature_map_size(model, x[:1])
        res["input_side"] = int(x.shape[-1])
        d = {f: {m: [] for m in (*METHODS, "random")} for f in fills}
        for i in idx:
            xi = x[i:i + 1]
            with torch.no_grad():
                t = int(model(xi).argmax(1))
            maps = {m: explain(model, xi, m, target=t) for m in METHODS}
            for f in fills:
                d[f]["random"].append(random_deletion_auc(model, xi, target=t, seed=int(i), fill=f))
                for m, h in maps.items():
                    d[f][m].append(deletion_auc(model, xi, h, target=t, fill=f))
        res["per_seed"].append(d)
        print(f"  done {c}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True,
                    help="name=ckpt1,ckpt2,ckpt3@data.npz")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--fills", nargs="+", default=["zero", "blur"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = resolve_device(a.device)
    n_val = len(load_npz(a.config[0].split("@")[1])["val"][1])
    # same selection rule as scripts/explain.py (seed 0, without replacement)
    idx = np.sort(np.random.default_rng(0).choice(n_val, size=min(a.n_images, n_val), replace=False))

    raw = {}
    for spec in a.config:
        name, rest = spec.split("=", 1)
        ck, data = rest.split("@")
        print(f"config {name}", flush=True)
        raw[name] = run_config(ck.split(","), data, idx, a.fills, dev)

    # per-image gaps averaged over seeds
    gaps = {}
    for name, r in raw.items():
        gaps[name] = {}
        for f in a.fills:
            g = {}
            for m in METHODS:
                per_seed = [np.array(s[f]["random"]) - np.array(s[f][m]) for s in r["per_seed"]]
                g[m] = np.mean(per_seed, 0)
                g[m + "_per_seed_mean"] = [float(v.mean()) for v in per_seed]
            gaps[name][f] = g

    summary = {"val_indices": idx.tolist(), "n_random_orders": 5, "deletion_steps": 20,
               "fills": a.fills, "configs": {}, "contrasts": {}}
    for name, r in raw.items():
        c = {"feature_map": r["feature_map"], "input_side": r["input_side"], "n_seeds": len(r["ckpts"]),
             "ckpts": r["ckpts"], "fills": {}}
        for f in a.fills:
            c["fills"][f] = {"random_deletion_auc": float(np.mean([np.mean(s[f]["random"]) for s in r["per_seed"]]))}
            for m in METHODS:
                c["fills"][f][m] = {"deletion_auc": float(np.mean([np.mean(s[f][m]) for s in r["per_seed"]])),
                                   "gap_vs_random": boot_ci(gaps[name][f][m]),
                                   "gap_per_seed": gaps[name][f][m + "_per_seed_mean"]}
        summary["configs"][name] = c

    pairs = []
    names = list(raw)
    for p, q in (("cnn64", "cnn28"), ("res64", "res28"), ("res28", "cnn28"), ("res64", "cnn64")):
        if p in names and q in names:
            pairs.append((p, q))
    for p, q in pairs:
        summary["contrasts"][f"{p}-{q}"] = {f: {m: boot_ci(gaps[p][f][m] - gaps[q][f][m]) for m in METHODS}
                                            for f in a.fills}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    raw_out = os.path.splitext(a.out)[0] + "_per_image.json"
    with open(raw_out, "w") as fh:
        json.dump(raw, fh)
    print(json.dumps({k: {f: {m: v["fills"][f][m]["gap_vs_random"] for m in METHODS} for f in a.fills}
                      for k, v in summary["configs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
