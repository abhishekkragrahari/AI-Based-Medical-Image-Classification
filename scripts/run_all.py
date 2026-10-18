"""Run the whole pipeline in order: download -> E0 validate -> E2-E4 (+E1) -> E5 explain
-> E7 final test (only with --final) -> Markdown report.

Each step calls the same script you can run by hand, so outputs are identical.
--quick is a fast smoke run (1 seed, 2 epochs) whose outputs go to runs/quick and
reports/quick so they never mix with the real results.

Usage:
  python -m scripts.run_all --quick              # check everything works (~1 min CPU)
  python -m scripts.run_all                      # E0, E2-E4, E5 on val; test untouched
  python -m scripts.run_all --final              # same, then E7 once and README results
  python -m scripts.run_all --final --include-e1 # also the ~4 h CPU ResNet-18 protocol
"""

import argparse
import json
import os
import sys

from scripts import download_data, explain, final_eval, make_report, run_experiments, validate_data

CONFIGS = ["configs/small_cnn.yaml", "configs/resnet18.yaml",
           "configs/small_cnn_classweight.yaml", "configs/small_cnn_noaug.yaml"]
E1_CONFIG = "configs/resnet18_medmnist_protocol.yaml"


def _step(name, rc):
    print(f"--- {name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, separate output folders")
    ap.add_argument("--final", action="store_true", help="also run the one-time held-out test evaluation")
    ap.add_argument("--include-e1", action="store_true", help="also run the long MedMNIST ResNet-18 protocol")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--runs-dir")
    ap.add_argument("--reports-dir")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip download and the official split-size check (for tests on synthetic files)")
    a = ap.parse_args(argv)
    seeds = a.seeds or ([0] if a.quick else [0, 1, 2])
    epochs = a.epochs or (2 if a.quick else None)
    runs = a.runs_dir or ("runs/quick" if a.quick else "runs")
    reports = a.reports_dir or ("reports/quick" if a.quick else "reports")
    os.makedirs(reports, exist_ok=True)
    common = ["--seeds", *map(str, seeds), "--data-path", a.data, "--out-dir", runs]
    if epochs:
        common += ["--epochs", str(epochs)]

    if not a.synthetic:
        if _step("download", download_data.main(["--root", os.path.dirname(a.data) or "."])):
            return 1
    v_args = ["--data", a.data, "--out", os.path.join(reports, "data_validation_28.json")]
    if _step("E0 validate", validate_data.main(v_args + (["--no-official-counts"] if a.synthetic else []))):
        print("data validation failed; fix the data before training", file=sys.stderr)
        return 1

    val_report = os.path.join(reports, "val_summary.json")
    if _step("E2-E4 train", run_experiments.main(["--configs", *a.configs, "--report", val_report] + common)):
        return 1
    if a.include_e1:
        if _step("E1 train", run_experiments.main(["--configs", E1_CONFIG, "--report",
                                                   os.path.join(reports, "val_e1.json")] + common)):
            return 1

    with open(val_report) as f:
        val = json.load(f)
    sel = val["_selected"]
    ckpts = [os.path.join(r, "best.pt") for r in val[sel]["runs"]]
    x_args = ["--ckpts", *ckpts, "--data", a.data, "--split", "val",
              "--out", os.path.join(reports, f"xai_{sel}_val")]
    if a.quick:
        x_args += ["--n-per-group", "1", "--n-stats", "10"]
    if _step("E5 explain", explain.main(x_args)):
        return 1

    if a.final:
        if _step("E7 final test", final_eval.main(["--val-report", val_report, "--data", a.data, "--final",
                                                   "--out", os.path.join(reports, "test_summary.json")]
                                                  + (["--n-boot", "100"] if a.quick else []))):
            return 1
    else:
        print("--- E7 final test: skipped (add --final when you are ready to look at the test set once)")

    # Only the real, final run writes into README; quick/synthetic runs stay in their report folder.
    update = a.final and not a.quick and not a.synthetic
    return _step("report", make_report.main(["--reports", reports] + (["--update-readme"] if update else [])))


if __name__ == "__main__":
    sys.exit(main())
