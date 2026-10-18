"""E7: one-time held-out test evaluation of every run listed in a validation report.

All runs (not just the selected one) are reported, so the test set is not used to
pick a winner. The selected configuration comes from the validation report.
Usage: python -m scripts.final_eval --val-report reports/val_summary.json --final
"""

import argparse
import json
import os
import sys

import numpy as np

from medxai.evaluate import evaluate_checkpoint

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
    for name, entry in val.items():
        if name.startswith("_"):
            continue
        runs = [evaluate_checkpoint(os.path.join(r, "best.pt"), a.data, "test", final=True,
                                    n_boot=a.n_boot) for r in entry["runs"]]
        agg = {}
        for at in ("at_0.5", "at_val_youden"):
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
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({n: m["aggregate"]["at_0.5"]["auroc"] for n, m in out["models"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
