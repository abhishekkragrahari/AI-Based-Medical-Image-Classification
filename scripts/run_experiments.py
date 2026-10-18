"""Run a config x seed matrix and aggregate *validation* metrics (mean, std).

Model selection rule (fixed in EXPERIMENT_PLAN.md): highest mean val AUROC.
Usage: python -m scripts.run_experiments --configs configs/small_cnn.yaml configs/resnet18.yaml --seeds 0 1 2
"""

import argparse
import json
import os
import sys

import numpy as np

from medxai.train import TrainConfig, train

KEYS = ("auroc", "auprc", "balanced_accuracy", "accuracy", "macro_f1", "ece")


def aggregate(summaries: list[dict], at: str = "val_at_0.5") -> dict:
    out = {}
    for k in KEYS:
        v = np.array([s[at][k] for s in summaries], dtype=float)
        out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  "values": v.tolist()}
    for cls, key in (("pneumonia", "recall(sensitivity)"), ("normal", "recall(specificity)")):
        v = np.array([s[at]["per_class"][cls][key] for s in summaries], dtype=float)
        out[f"{cls}_{key}"] = {"mean": float(np.mean(v)),
                               "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--data-path")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--report", default="reports/val_summary.json")
    a = ap.parse_args(argv)
    report = {}
    for cfg_path in a.configs:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        sums = []
        for seed in a.seeds:
            cfg = TrainConfig.from_file(cfg_path, seed=seed, data_path=a.data_path, epochs=a.epochs,
                                        out_dir=a.out_dir, run_name=f"{name}_s{seed}")
            print(f"=== {name} seed={seed}", flush=True)
            sums.append(train(cfg, verbose=True))
        report[name] = {
            "config": cfg_path, "seeds": a.seeds, "runs": [s["run_dir"] for s in sums],
            "best_epochs": [s["best_epoch"] for s in sums],
            "train_time_s": [s["train_time_s"] for s in sums], "n_params": sums[0]["n_params"],
            "val_at_0.5": aggregate(sums, "val_at_0.5"),
            "val_at_youden": aggregate(sums, "val_at_youden"),
        }
    report["_selection_rule"] = "highest mean val AUROC across seeds"
    report["_selected"] = max((k for k in report if not k.startswith("_")),
                              key=lambda k: report[k]["val_at_0.5"]["auroc"]["mean"])
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: (v["val_at_0.5"]["auroc"] if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
