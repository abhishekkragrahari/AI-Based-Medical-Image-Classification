"""Evaluate a checkpoint. Test split requires --final (E7 only).

Usage: python -m scripts.evaluate --ckpt runs/<run>/best.pt --split val
"""

import argparse
import json
import sys

from medxai.evaluate import evaluate_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/pneumoniamnist.npz")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the held-out test split")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    try:
        res = evaluate_checkpoint(a.ckpt, a.data, a.split, final=a.final, n_boot=a.n_boot)
    except PermissionError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
