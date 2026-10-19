"""Paired comparison of two final test evaluations (e.g. 28 px vs 64 px) on the same test images.

Reads the seed-ensemble scores that final_eval writes to test_scores.json and the
test labels from the data file, then reports a paired bootstrap of the AUROC
difference and both sides' metrics at threshold 0.5. The two data files must hold
the same test images in the same order (checked through their labels).

Usage: python -m scripts.compare_sizes --a reports/test_scores.json --a-data data/pneumoniamnist.npz
       --b reports/size64/test_scores.json --b-data data/pneumoniamnist_64.npz
       [--model small_cnn] [--out reports/compare_28_vs_64.json]
"""

import argparse
import json
import os
import sys

import numpy as np

from medxai.data import load_npz
from medxai.metrics import binary_metrics, paired_bootstrap_auroc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-data", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-data", required=True)
    ap.add_argument("--model", default="small_cnn")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="reports/compare_28_vs_64.json")
    a = ap.parse_args(argv)
    ya, yb = load_npz(a.a_data)["test"][1], load_npz(a.b_data)["test"][1]
    if not np.array_equal(ya, yb):
        print("the two test splits differ (labels not identical); a paired comparison is not valid",
              file=sys.stderr)
        return 1
    pa = np.array(json.load(open(a.a))["scores"][a.model])
    pb = np.array(json.load(open(a.b))["scores"][a.model])
    out = {"model": a.model, "a": a.a, "b": a.b, "n_test": int(len(ya)),
           "a_at_0.5": binary_metrics(ya, pa, 0.5), "b_at_0.5": binary_metrics(ya, pb, 0.5),
           "b_minus_a": paired_bootstrap_auroc(ya, pb, pa, n_boot=a.n_boot)}
    # Paired bootstrap of the specificity and balanced-accuracy differences at 0.5.
    rng = np.random.default_rng(0)
    d = {"specificity": [], "balanced_accuracy": []}
    for _ in range(a.n_boot):
        i = rng.integers(0, len(ya), len(ya))
        if len(np.unique(ya[i])) < 2:
            continue
        ma, mb = binary_metrics(ya[i], pa[i], 0.5), binary_metrics(ya[i], pb[i], 0.5)
        d["specificity"].append(mb["per_class"]["normal"]["recall(specificity)"]
                                - ma["per_class"]["normal"]["recall(specificity)"])
        d["balanced_accuracy"].append(mb["balanced_accuracy"] - ma["balanced_accuracy"])
    out["b_minus_a_at_0.5_ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                    for k, v in d.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
