"""E0: dataset integrity and leakage audit. Writes a JSON report.

Usage: python -m scripts.validate_data --data data/pneumoniamnist.npz --out reports/data_validation_28.json
"""

import argparse
import json
import os
import sys

from medxai.data import DATASETS, dataset_info, load_npz, md5sum, validate_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--dataset", default="pneumoniamnist", choices=sorted(DATASETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-official-counts", action="store_true",
                    help="skip the check against official MedMNIST split sizes (synthetic data)")
    a = ap.parse_args(argv)
    info = dataset_info(a.dataset)
    rep = validate_dataset(load_npz(a.data), None if a.no_official_counts else info["counts"],
                           n_classes=len(info["classes"]))
    md5 = md5sum(a.data)
    result = {"file": os.path.basename(a.data), "dataset": a.dataset, "md5": md5,
              "md5_matches_official": md5 in info["md5"].values(),
              "ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings, "stats": rep.stats}
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
