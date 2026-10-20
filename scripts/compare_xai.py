"""E9: paired comparison of explanation faithfulness (and accuracy) between training methods.

Every group is a set of checkpoints (one per seed) trained with one method. All groups
are scored on the same images of one split (validation by default; the test split needs
--final). Per image and checkpoint it computes the deletion AUC of each XAI method and of
random deletion; the "gap" is random minus method (higher = the map ranks pixels the model
relies on better than chance). Gaps are averaged over seeds, then each group is compared
with the reference group by a paired bootstrap over images. Seed-ensemble AUROC and
metrics at 0.5 are reported too, with a paired bootstrap of the AUROC difference.

Usage:
  python -m scripts.compare_xai --group baseline=runs/e9/small_cnn_s0/best.pt,runs/e9/small_cnn_s1/best.pt \
      --group cfr=runs/e9/small_cnn_cfr_s0/best.pt,... --ref baseline --out reports/e9/xai_compare_val.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from medxai.data import load_npz
from medxai.metrics import binary_metrics, paired_bootstrap_auroc
from medxai.models import load_checkpoint
from medxai.train import predict_proba
from medxai.xai import border_mass, deletion_auc, explain, random_deletion_auc
from scripts.explain import to_tensor

METHODS = ("gradcam", "ig", "occlusion")


def paired_bootstrap_mean(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """mean(a - b) over paired cases with a percentile 95% CI from resampling cases."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                   float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, p)), "n": int(len(d))}


def score_checkpoint(path, imgs, idx, steps: int = 20, n_random: int = 5) -> tuple[dict, np.ndarray]:
    model, meta = load_checkpoint(path)
    x = to_tensor(imgs, meta["normalizer"])
    out = {m: {"deletion_auc": [], "border_mass": []} for m in METHODS}
    out["random"] = {"deletion_auc": []}
    for i in idx:
        xi = x[i : i + 1]
        with torch.no_grad():
            t = int(model(xi).argmax(1))
        for m in METHODS:
            h = explain(model, xi, m, target=t)
            out[m]["deletion_auc"].append(deletion_auc(model, xi, h, target=t, steps=steps))
            out[m]["border_mass"].append(border_mass(h))
        out["random"]["deletion_auc"].append(random_deletion_auc(model, xi, target=t, steps=steps,
                                                                 n=n_random, seed=int(i)))
    return out, predict_proba(model, x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", action="append", required=True, help="name=ckpt[,ckpt...]")
    ap.add_argument("--ref", required=True, help="reference group name")
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n-images", type=int, default=0, help="0 = every image of the split")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing to touch the test split without --final", file=sys.stderr)
        return 2
    groups = {}
    for g in a.group:
        name, _, paths = g.partition("=")
        groups[name] = [p for p in paths.split(",") if p]
    if a.ref not in groups:
        print(f"--ref {a.ref} is not one of the groups {sorted(groups)}", file=sys.stderr)
        return 2
    imgs, y = load_npz(a.data)[a.split]
    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(y)) if not a.n_images or a.n_images >= len(y) else \
        np.sort(rng.choice(len(y), a.n_images, replace=False))

    per_image, scores = {}, {}
    for name, ckpts in groups.items():
        runs = [score_checkpoint(c, imgs, idx) for c in ckpts]
        scores[name] = np.mean([p for _, p in runs], axis=0)
        per_image[name] = {m: {k: np.mean([r[m][k] for r, _ in runs], axis=0) for k in runs[0][0][m]}
                           for m in runs[0][0]}
        for m in METHODS:
            per_image[name][m]["gap"] = per_image[name]["random"]["deletion_auc"] - per_image[name][m]["deletion_auc"]

    def summ(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "std": float(v.std())}

    report = {"split": a.split, "n_images": int(len(idx)), "ref": a.ref,
              "groups": {n: {"checkpoints": c} for n, c in groups.items()},
              "notes": ["deletion fill is 0 (training mean); CFR trains with a blurred fill",
                        "gap = random_deletion_auc - method deletion_auc, averaged over seeds; higher is better",
                        "classification: seed-ensemble (mean probability) on the same split"]}
    for name in groups:
        g = report["groups"][name]
        g["xai"] = {m: {k: summ(v) for k, v in d.items()} for m, d in per_image[name].items()}
        g["ensemble_at_0.5"] = binary_metrics(y, scores[name], 0.5)
        if name != a.ref:
            g["vs_ref"] = {"auroc": paired_bootstrap_auroc(y, scores[name], scores[a.ref], a.n_boot, a.seed)}
            for m in METHODS:
                g["vs_ref"][f"{m}_gap"] = paired_bootstrap_mean(per_image[name][m]["gap"],
                                                                per_image[a.ref][m]["gap"],
                                                                a.n_boot, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    for name, g in report["groups"].items():
        print(name, "auroc", round(g["ensemble_at_0.5"]["auroc"], 4),
              {m: round(g["xai"][m]["gap"]["mean"], 3) for m in METHODS}, g.get("vs_ref", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
