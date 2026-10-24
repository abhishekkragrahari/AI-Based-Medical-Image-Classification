"""E5: generate explanations and explanation-quality statistics.

Runs on the validation split by default; the test split needs --final.
Samples are chosen per outcome (TP/TN/FP/FN for binary; correct/wrong for multi-class)
and spread over confidence levels.
Sample identifiers are split indices only (no patient information exists in the data).

Faithfulness (E10): ``--fills`` and ``--units`` add deletion *and* insertion AUCs against
random attribution for each fill strategy (zero | blur | noise) and removal unit
(pixel | patch) under ``faithfulness``; ``--randomisation N`` runs the cascading model
randomisation sanity check on N images. The top-level ``deletion_auc`` /
``random_deletion_auc`` keys keep the E5 definition (zero fill, pixels).

Usage:
  python -m scripts.explain --ckpts runs/resnet18_s0/best.pt runs/resnet18_s1/best.pt \
      --split val --n-per-group 4 --out reports/xai_resnet18_val
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from medxai import CLASS_NAMES
from medxai.data import load_npz, to_model_input
from medxai.models import load_checkpoint
from medxai.train import predict_proba_all, resolve_device
from medxai.xai import (FILLS, UNITS, border_mass, deletion_auc, explain, insertion_auc,
                        model_randomisation_test, overlay, perturbation_stability,
                        random_deletion_auc, random_insertion_auc, rank_agreement, topk_iou)

METHODS = ("gradcam", "ig", "occlusion")


def to_tensor(imgs, norm):
    return to_model_input(imgs, norm)


def pick_samples_multiclass(y, proba, n_per_group, seed=0):
    """Correct / wrong predictions, each spread over top-label confidence."""
    pred, conf = proba.argmax(1), proba.max(1)
    chosen = {}
    for g, m in (("correct", pred == y), ("wrong", pred != y)):
        idx = np.where(m)[0]
        idx = idx[np.argsort(conf[idx])]
        k = min(n_per_group, len(idx))
        chosen[g] = sorted(set(idx[np.linspace(0, len(idx) - 1, k).round().astype(int)].tolist())) if k else []
    return chosen


def pick_samples(y, p, n_per_group, seed=0):
    rng = np.random.default_rng(seed)
    pred = (p >= 0.5).astype(int)
    groups = {"TP": (y == 1) & (pred == 1), "TN": (y == 0) & (pred == 0),
              "FP": (y == 0) & (pred == 1), "FN": (y == 1) & (pred == 0)}
    chosen = {}
    for g, m in groups.items():
        idx = np.where(m)[0]
        if len(idx) == 0:
            chosen[g] = []
            continue
        # spread over confidence: sort by confidence and take evenly spaced quantiles
        conf = np.abs(p[idx] - 0.5)
        idx = idx[np.argsort(conf)]
        k = min(n_per_group, len(idx))
        sel = idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]
        chosen[g] = sorted(set(sel.tolist()))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True, help="first = primary; others for cross-seed agreement")
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n-per-group", type=int, default=4)
    ap.add_argument("--n-stats", type=int, default=100, help="samples for aggregate statistics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fills", nargs="+", default=["zero"], choices=FILLS)
    ap.add_argument("--units", nargs="+", default=["pixel"], choices=UNITS)
    ap.add_argument("--randomisation", type=int, default=0,
                    help="images for the cascading model-randomisation check (0 = skip)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        print("refusing to touch the test split without --final", file=sys.stderr)
        return 2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    dev = resolve_device(a.device)
    models = [(m.to(dev), meta) for m, meta in (load_checkpoint(c) for c in a.ckpts)]
    model, meta = models[0]
    names = meta.get("class_names", list(CLASS_NAMES))
    imgs, y = load_npz(a.data)[a.split]
    x = to_tensor(imgs, meta["normalizer"]).to(dev)
    proba = predict_proba_all(model, x)
    binary = proba.shape[1] == 2
    p = proba[:, 1] if binary else proba.max(1)

    # ---- qualitative figure -------------------------------------------------
    if binary:
        chosen = pick_samples(y, p, a.n_per_group, a.seed)
        groups = ("TP", "TN", "FP", "FN")
    else:
        chosen = pick_samples_multiclass(y, proba, a.n_per_group, a.seed)
        groups = ("correct", "wrong")
    rows = [(g, i) for g in groups for i in chosen[g]]
    if rows:
        fig, axes = plt.subplots(len(rows), 1 + len(METHODS), figsize=(2.2 * (1 + len(METHODS)), 2.2 * len(rows)))
        axes = np.atleast_2d(axes)
        for r, (g, i) in enumerate(rows):
            xi = x[i : i + 1]
            img01 = imgs[i] / 255.0
            axes[r, 0].imshow(img01, cmap="gray" if img01.ndim == 2 else None, vmin=0, vmax=1)
            score = f"p({names[1][:4]})={p[i]:.2f}" if binary else f"pred={names[int(proba[i].argmax())][:10]} {p[i]:.2f}"
            axes[r, 0].set_title(f"{a.split}#{i} {g}\ny={names[y[i]][:12]} {score}", fontsize=7)
            for c, m in enumerate(METHODS, start=1):
                h = explain(model, xi, m)  # explains the predicted class
                axes[r, c].imshow(overlay(img01, h))
                axes[r, c].set_title(f"{m} border={border_mass(h):.2f}", fontsize=7)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle("Explanations of the predicted class (research use only; not clinical evidence)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "examples.png"), dpi=130)
        plt.close(fig)

    # ---- quantitative checks ------------------------------------------------
    rng = np.random.default_rng(a.seed)
    stat_idx = np.sort(rng.choice(len(y), size=min(a.n_stats, len(y)), replace=False))
    per = {m: {"border_mass": [], "deletion_auc": [], "stability": []} for m in METHODS}
    random_del, cross_method, cross_seed = [], {}, {m: [] for m in METHODS}
    combos = [(f, u) for f in a.fills for u in a.units]
    faith = {f"{f}/{u}": {**{m: {"deletion_auc": [], "insertion_auc": [], "deletion_gap": [],
                                 "insertion_gap": []} for m in METHODS},
                          "random": {"deletion_auc": [], "insertion_auc": []}} for f, u in combos}
    for i in stat_idx:
        xi = x[i : i + 1]
        with torch.no_grad():
            t = int(model(xi).argmax(1))
        maps = {m: explain(model, xi, m, target=t) for m in METHODS}
        for m, h in maps.items():
            per[m]["border_mass"].append(border_mass(h))
            per[m]["deletion_auc"].append(deletion_auc(model, xi, h, target=t))
        random_del.append(random_deletion_auc(model, xi, target=t, seed=int(i)))
        for f, u in combos:
            d = faith[f"{f}/{u}"]
            rd = random_deletion_auc(model, xi, target=t, seed=int(i), fill=f, unit=u)
            ri = random_insertion_auc(model, xi, target=t, seed=int(i), fill=f, unit=u)
            d["random"]["deletion_auc"].append(rd)
            d["random"]["insertion_auc"].append(ri)
            for m, h in maps.items():
                da = deletion_auc(model, xi, h, target=t, fill=f, unit=u)
                ia = insertion_auc(model, xi, h, target=t, fill=f, unit=u)
                d[m]["deletion_auc"].append(da)
                d[m]["insertion_auc"].append(ia)
                d[m]["deletion_gap"].append(rd - da)  # > 0: better than random
                d[m]["insertion_gap"].append(ia - ri)  # > 0: better than random
        for j, m1 in enumerate(METHODS):
            for m2 in METHODS[j + 1 :]:
                cross_method.setdefault(f"{m1}~{m2}", []).append(rank_agreement(maps[m1], maps[m2]))
        for other, _ in models[1:]:
            for m in METHODS:
                cross_seed[m].append(topk_iou(maps[m], explain(other, xi, m, target=t), 0.1))
    # stability is costlier: use a subset
    for i in stat_idx[: min(30, len(stat_idx))]:
        for m in METHODS:
            per[m]["stability"].append(perturbation_stability(model, x[i : i + 1], m, sigma=0.05, n=3,
                                                              seed=int(i)))
    randomisation = None
    if a.randomisation:
        stages = {m: {} for m in METHODS}
        for i in stat_idx[: a.randomisation]:
            for m in METHODS:
                for r in model_randomisation_test(model, x[i : i + 1], m, seed=int(i)):
                    s = stages[m].setdefault(r["randomised_through"], {"spearman": [], "top10pct_iou": []})
                    s["spearman"].append(r["spearman"])
                    s["top10pct_iou"].append(r["top10pct_iou"])

    def ms(v):
        v = np.asarray(v, dtype=float)
        ok = v[np.isfinite(v)]
        if not len(ok):
            return {"mean": None, "std": None, "n": 0, "n_undefined": int(len(v))}
        return {"mean": float(ok.mean()), "std": float(ok.std()), "n": int(len(ok)),
                "n_undefined": int(len(v) - len(ok))}

    if a.randomisation:
        randomisation = {m: {st: {k: ms(v) for k, v in d.items()} for st, d in sd.items()}
                         for m, sd in stages.items()}

    stats = {
        "split": a.split, "checkpoints": a.ckpts, "n_samples_stats": int(len(stat_idx)),
        "chosen_examples": chosen,
        "uniform_border_mass_reference": border_mass(np.ones(imgs.shape[1:3])),
        "per_method": {m: {k: ms(v) for k, v in d.items()} for m, d in per.items()},
        "random_deletion_auc": ms(random_del),
        "cross_method_spearman": {k: ms(v) for k, v in cross_method.items()},
        "cross_seed_top10pct_iou": {m: ms(v) for m, v in cross_seed.items()} if len(models) > 1 else None,
        "input_size": int(imgs.shape[1]), "class_names": names,
        "faithfulness": {k: {m: {kk: ms(vv) for kk, vv in d.items()} for m, d in fd.items()}
                         for k, fd in faith.items()},
        "model_randomisation": randomisation,
        "notes": ["deletion_auc: lower than random_deletion_auc => attribution ranks pixels the model relies on",
                  "border_mass well above uniform_border_mass_reference suggests reliance on image borders (possible shortcut)",
                  "stability: Spearman between maps for x and x+N(0,0.05^2) in normalised units",
                  "faithfulness[fill/unit]: deletion_gap = random - method deletion AUC, insertion_gap = "
                  "method - random insertion AUC; > 0 means better than random attribution",
                  "model_randomisation: Spearman/IoU vs the trained model's map after re-initialising blocks "
                  "from the output down; high values = insensitive to learned weights (fails the sanity check)"],
    }
    with open(os.path.join(a.out, "xai_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({m: {k: (v["mean"] if v else None) for k, v in d.items()}
                      for m, d in stats["per_method"].items()}, indent=1))
    print("random_deletion_auc", stats["random_deletion_auc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
