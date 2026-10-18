"""Train one model. Usage: python -m scripts.train --config configs/small_cnn.yaml --seed 0"""

import argparse
import json
import sys

from medxai.train import TrainConfig, train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--data-path")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--max-train-samples", type=int)
    a = ap.parse_args(argv)
    cfg = TrainConfig.from_file(a.config, seed=a.seed, data_path=a.data_path, epochs=a.epochs,
                                out_dir=a.out_dir, run_name=a.run_name,
                                max_train_samples=a.max_train_samples)
    s = train(cfg)
    print(json.dumps({k: s[k] for k in ("run_dir", "best_epoch", "train_time_s", "n_params")}))
    print(json.dumps({"val_auroc": s["val_at_0.5"]["auroc"],
                      "val_bal_acc@0.5": s["val_at_0.5"]["balanced_accuracy"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
