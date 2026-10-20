"""E7: one-time held-out test evaluation of every run listed in a validation report.

All runs (not just the selected one) are reported, so the test set is not used to
pick a winner. The selected configuration comes from the validation report.
Each configuration's seeds are also averaged into one ensemble score per test image;
those scores (no labels) go to <out dir>/test_scores.json, and every configuration's
ensemble is compared with the selected one by a paired bootstrap of the AUROC difference.
Usage: python -m scripts.final_eval --val-report reports/val_summary.json --final
"""

import argparse
import json
import os
import sys

import numpy as np

from medxai.data import load_npz, md5sum
from medxai.evaluate import evaluate_checkpoint
from medxai.metrics import binary_metrics, paired_bootstrap_auroc

KEYS = ("auroc", "auprc", "balanced_accuracy", "accuracy", "macro_f1", "ece")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-report", default="reports/val_summary.json")
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--out", default="reports/test_summary.json")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    if not a.final:
        print("refusing to touch the test split without --final", file=sys.stderr)
        return 2
    with open(a.val_report) as f:
        val = json.load(f)
    out = {"selected_by_validation": val["_selected"], "models": {}}
    ens = {}
    for name, entry in val.items():
        if name.startswith("_"):
            continue
        runs = [evaluate_checkpoint(os.path.join(r, "best.pt"), a.data, "test", final=True,
                                    n_boot=a.n_boot) for r in entry["runs"]]
        agg = {}
        for at in ("at_0.5", "at_val_youden", "at_val_youden_dedup"):
            if not all(at in r for r in runs):
                continue
            agg[at] = {}
            for k in KEYS:
                v = np.array([r[at][k] for r in runs], dtype=float)
                agg[at][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                              "values": v.tolist()}
            for cls, key in (("pneumonia", "recall(sensitivity)"), ("normal", "recall(specificity)")):
                v = np.array([r[at]["per_class"][cls][key] for r in runs], dtype=float)
                agg[at][f"{cls}_{key}"] = {"mean": float(v.mean()),
                                           "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        out["models"][name] = {"runs": entry["runs"], "aggregate": agg,
                               "per_run_ci95_at_0.5": [r["ci95_at_0.5"] for r in runs],
                               "per_run_confusion_at_0.5": [r["at_0.5"]["confusion"] for r in runs]}
        ens[name] = np.mean([np.load(os.path.join(r, "test_scores.npy")) for r in entry["runs"]], axis=0)
    labels = load_npz(a.data)["test"][1]
    sel = out["selected_by_validation"]
    for name, p in ens.items():
        out["models"][name]["seed_ensemble_at_0.5"] = binary_metrics(labels, p, 0.5)
        if name != sel:
            out["models"][name]["paired_auroc_vs_selected"] = paired_bootstrap_auroc(
                labels, p, ens[sel], n_boot=max(a.n_boot, 200))
    with open(os.path.join(os.path.dirname(a.out) or ".", "test_scores.json"), "w") as f:
        json.dump({"data_md5": md5sum(a.data), "note": "seed-ensemble P(pneumonia) per test image, "
                   "in file order; labels are not stored", "scores": {k: np.round(v, 6).tolist()
                                                                      for k, v in ens.items()}}, f)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({n: m["aggregate"]["at_0.5"]["auroc"] for n, m in out["models"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
