"""Paper Figure 9: per-epoch training loss and validation AUROC of the small CNN.

Reads the per-epoch records in reports/training_curves/ (GPU re-runs with the reported
configuration, seeds 0-2, 28 and 64 px; no test data were used) and writes
reports/training_curves/fig_training_curves.png.

Records were produced with:
  python -m scripts.train --config configs/small_cnn.yaml --seed S --out-dir runs/gpu/s28
  python -m scripts.train --config configs/small_cnn.yaml --seed S --data-path data/pneumoniamnist_64.npz --out-dir runs/gpu/s64
(with device: cuda). Usage: python -m scripts.plot_training_curves [--out PATH]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "training_curves"
COLOURS = {"28": "#2a78d6", "64": "#e8743b"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SRC / "fig_training_curves.png"))
    a = ap.parse_args(argv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for px, c in COLOURS.items():
        for s in range(3):
            h = json.loads((SRC / f"small_cnn_{px}px_s{s}.json").read_text())
            ep = [r["epoch"] for r in h]
            lab = f"{px} px" if s == 0 else None
            ax1.plot(ep, [r["train_loss"] for r in h], color=c, lw=1.2, label=lab)
            ax2.plot(ep, [r["val_auroc"] for r in h], color=c, lw=1.2, label=lab)
    ax1.set(xlabel="Epoch", ylabel="Training loss")
    ax2.set(xlabel="Epoch", ylabel="Validation AUROC")
    ax2.legend(title="Small CNN, 3 seeds each", frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
