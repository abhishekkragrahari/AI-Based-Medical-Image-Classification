"""Build a validation report (same format as run_experiments) from finished run folders.

Used when the seeds of one configuration were trained in separate jobs.
Usage: python -m scripts.aggregate_runs --runs runs/e1/resnet18_medmnist_protocol_s0 ... --name resnet18_medmnist_protocol
       --config configs/resnet18_medmnist_protocol.yaml --report reports/e1/val_e1.json
"""

import argparse
import json
import os
import sys

from scripts.run_experiments import aggregate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    sums = []
    for r in sorted(a.runs):
        with open(os.path.join(r, "summary.json")) as f:
            sums.append(json.load(f))
    entry = {"config": a.config, "seeds": [s["config"]["seed"] for s in sums], "runs": sorted(a.runs),
             "best_epochs": [s["best_epoch"] for s in sums], "train_time_s": [s["train_time_s"] for s in sums],
             "n_params": sums[0]["n_params"], "val_at_0.5": aggregate(sums, "val_at_0.5"),
             "val_at_youden": aggregate(sums, "val_at_youden")}
    if all(s.get("val_dedup") for s in sums):
        entry["val_dedup_at_0.5"] = aggregate([{"m": s["val_dedup"]["at_0.5"]} for s in sums], "m")
        entry["val_dedup_n"] = sums[0]["val_dedup"]["n"]
    report = {a.name: entry, "_selection_rule": "single configuration", "_selected": a.name}
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(entry["val_at_0.5"]["auroc"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
