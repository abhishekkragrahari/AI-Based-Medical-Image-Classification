"""Draw the manuscript figures from the result files in reports/ (see reports/MANIFEST.md).

Figures 1-5 and 7-8 of the manuscript, plus the controlled saliency figure added in
revision 1 (drawn only if reports/revision/controlled_xai.json exists). Figure 6 is the
example grid that scripts/explain.py writes, and Figure 9 is drawn by
scripts/plot_training_curves.py.

Test labels are read from the official PneumoniaMNIST files; per-image predictions are the
saved seed-ensemble probabilities. Usage:
  python -m scripts.make_paper_figures [--out reports/figures] [--data data/pneumoniamnist.npz]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from medxai.data import load_npz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
BLUE, ORANGE, GREEN = "#2a78d6", "#e8743b", "#2e9e6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = None


def jload(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def per_seed_auroc(summary, model):
    return jload(summary)["models"][model]["aggregate"]["at_0.5"]["auroc"]["values"]


def fig_model_comparison():
    rows = [("ResNet-18, benchmark recipe", "e1/test_summary.json", "resnet18_medmnist_protocol"),
            ("ResNet-18, our recipe", "test_summary.json", "resnet18"),
            ("Small CNN", "test_summary.json", "small_cnn"),
            ("Small CNN, class-weighted", "test_summary.json", "small_cnn_classweight"),
            ("Small CNN, no augmentation", "test_summary.json", "small_cnn_noaug"),
            ("Small CNN, 64 px †", "size64/test_summary.json", "small_cnn")]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    for y, (label, f, m) in enumerate(rows[::-1]):
        v = per_seed_auroc(f, m)
        ax.scatter(v, [y] * len(v), s=12, color=BLUE, alpha=0.45, edgecolor="none")
        ax.scatter([np.mean(v)], [y], s=40, color=BLUE, zorder=3)
        ax.text(np.mean(v), y + 0.22, f"{np.mean(v):.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axvline(0.944, color=MUTED, ls="--", lw=1)
    ax.text(0.944, len(rows) - 0.4, "published ResNet-18 (28 px)", color=MUTED, ha="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(0.92, 0.995)
    ax.set_xlabel("Test AUROC (small dots: seeds; large dot: mean)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_model_comparison")


def fig_paired_differences():
    ts = jload("test_summary.json")["models"]
    rec = jload("compare_resnet18_recipe_vs_e1.json")["b_minus_a"]
    res = jload("compare_28_vs_64.json")["b_minus_a"]
    rows = [("Our recipe − benchmark recipe\n(ResNet-18)", rec["auroc_diff"], rec["ci95"])]
    for m, lab in (("small_cnn", "Small CNN − ResNet-18"), ("small_cnn_classweight", "Class weights − ResNet-18"),
                   ("small_cnn_noaug", "No augmentation − ResNet-18")):
        p = ts[m]["paired_auroc_vs_selected"]
        rows.append((lab, p["auroc_diff"], p["ci95"]))
    rows.append(("64 px − 28 px (small CNN) †", res["auroc_diff"], res["ci95"]))
    fig, ax = plt.subplots(figsize=(6.3, 2.2))
    for y, (lab, d, ci) in enumerate(rows[::-1]):
        ax.hlines(y, ci[0], ci[1], color=BLUE, lw=2)
        ax.scatter([d], [y], s=36, color=BLUE, zorder=3, edgecolor="white")
        ax.text(ci[1] + 0.001, y, f"{d:+.4f}", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("Difference in test AUROC (paired bootstrap, 95% CI)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_paired_differences")


def ensembles(y28):
    s28 = jload("test_scores.json")["scores"]
    s64 = jload("size64/test_scores.json")["scores"]
    e1 = jload("e1/test_scores.json")["scores"]
    return {"Small CNN, 28 px": np.array(s28["small_cnn"]), "ResNet-18, 28 px": np.array(s28["resnet18"]),
            "Small CNN, 64 px": np.array(s64["small_cnn"]),
            "ResNet-18, benchmark recipe": np.array(e1["resnet18_medmnist_protocol"])}


def confusion(y, p):
    pred = (p >= 0.5).astype(int)
    return np.array([[np.sum((y == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])


def fig_confusion(y, ens):
    names = ["Small CNN, 28 px", "ResNet-18, 28 px", "Small CNN, 64 px"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    for ax, n in zip(axes, names):
        c = confusion(y, ens[n])
        share = c / c.sum(1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{c[i, j]}\n({share[i, j]:.0%})", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else INK, fontsize=7)
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("Predicted")
        ax.set_title(n, fontsize=8, loc="left")
    axes[0].set_ylabel("True")
    save(fig, "fig_confusion")


def fig_classwise(y, ens):
    names = [("Small CNN, 28 px", BLUE), ("ResNet-18, 28 px", ORANGE), ("Small CNN, 64 px", GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharey=True)
    for ax, cls, title in ((axes[0], 0, "Class: normal"), (axes[1], 1, "Class: pneumonia")):
        for n, col in names:
            c = confusion(y, ens[n])
            tp = c[cls, cls]
            prec, rec = tp / c[:, cls].sum(), tp / c[cls].sum()
            f1 = 2 * prec * rec / (prec + rec)
            ax.scatter([prec, rec, f1], [2, 1, 0], color=col, s=20, label=n)
        ax.set_yticks([2, 1, 0], ["Precision", "Recall", "F1"])
        ax.set_xlabel("Score")
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="x", color=GRID)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig_classwise")


def fig_roc(y, ens):
    order = [("ResNet-18, benchmark recipe", BLUE), ("ResNet-18, 28 px", ORANGE),
             ("Small CNN, 28 px", GREEN), ("Small CNN, 64 px", "#e3a72f")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for n, col in order:
        p = ens[n]
        fpr, tpr, _ = roc_curve(y, p)
        c = confusion(y, p)
        pt = (c[0, 1] / c[0].sum(), c[1, 1] / c[1].sum())
        lab = f"{n} (AUROC {roc_auc_score(y, p):.3f})"
        for ax in axes:
            ax.plot(fpr, tpr, color=col, lw=1.3, label=lab if ax is axes[0] else None)
            ax.scatter(*pt, color=col, s=14, zorder=3, edgecolor="white")
    axes[0].plot([0, 1], [0, 1], color=GRID, ls="--", lw=1)
    axes[1].set(xlim=(0, 0.4), ylim=(0.8, 1.005))
    axes[0].set_title("Full curve", fontsize=8, loc="left")
    axes[1].set_title("Zoom: false-positive rate 0–0.4", fontsize=8, loc="left")
    for ax in axes:
        ax.set(xlabel="False-positive rate (1 − specificity)", ylabel="True-positive rate (sensitivity)")
        ax.grid(color=GRID, lw=0.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_roc")


def fig_deletion():
    panels = [("ResNet-18, 28 px", "xai_resnet18_val/xai_stats.json"),
              ("Small CNN, 64 px", "size64/xai_small_cnn_val/xai_stats.json")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), sharex=True)
    for ax, (title, f) in zip(axes, panels):
        s = jload(f)
        rnd = s["random_deletion_auc"]["mean"]
        vals = {lab: s["per_method"][m]["deletion_auc"]["mean"]
                for m, lab in (("gradcam", "Grad-CAM"), ("ig", "IG"), ("occlusion", "Occlusion"))}
        names = list(vals)[::-1]
        ax.axvline(rnd, color=MUTED, lw=1, ls="--")
        ax.text(rnd, len(names) - 0.05, "random", color=MUTED, ha="center", va="bottom", fontsize=7)
        ax.hlines(range(len(names)), rnd, [vals[n] for n in names], color=GRID, lw=2)
        ax.scatter([vals[n] for n in names], range(len(names)), s=36, color=BLUE, zorder=3, edgecolor="white")
        for yy, n in enumerate(names):
            ax.text(vals[n], yy + 0.22, f"{vals[n]:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names) + 0.25)
        ax.set_xlim(0.40, 0.80)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Deletion AUC (lower = more faithful)")
        ax.tick_params(axis="y", length=0)
    save(fig, "fig_deletion")


def fig_cfr():
    d = jload("e9/xai_compare_val.json")
    groups = [("cfr", "CFR (pre-registered)"), ("cfr_del", "Deletion loss only"), ("cfr_eq", "Equivariance loss only")]
    gaps = [("gradcam_gap", "Grad-CAM gap"), ("ig_gap", "IG gap"), ("occlusion_gap", "Occlusion gap")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.9), sharey=True)
    for ax, (key, title) in zip(axes, gaps):
        ax.axvline(0, color=MUTED, lw=1)
        for yy, (g, _) in enumerate(groups[::-1]):
            v = d["groups"][g]["vs_ref"][key]
            ax.hlines(yy, *v["ci95"], color=BLUE, lw=2)
            ax.scatter([v["mean_diff"]], [yy], s=36, color=BLUE, zorder=3, edgecolor="white")
        ax.set_yticks(range(len(groups)), [g[1] for g in groups[::-1]])
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Difference vs baseline")
        ax.set_xlim(-0.05, 0.11)
        ax.tick_params(axis="y", length=0)
    save(fig, "fig_cfr")


def fig_controlled():
    f = R / "revision" / "controlled_xai.json"
    if not f.exists():
        print("skip controlled figure: reports/revision/controlled_xai.json not found")
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    cfgs = [("cnn28", "Small CNN, 28 px"), ("cnn64", "Small CNN, 64 px"),
            ("res28", "ResNet-18, 28 px"), ("res64", "ResNet-18, 64 px")]
    methods = [("gradcam", "Grad-CAM", BLUE), ("ig", "IG", ORANGE), ("occlusion", "Occlusion", GREEN)]
    fills = d["fills"]
    fig, axes = plt.subplots(1, len(fills), figsize=(3.3 * len(fills), 2.4), sharey=True, squeeze=False)
    for ax, fill in zip(axes[0], fills):
        for i, (key, _) in enumerate(cfgs[::-1]):
            for j, (m, lab, col) in enumerate(methods):
                g = d["configs"][key]["fills"][fill][m]["gap_vs_random"]
                y = i + (j - 1) * 0.22
                ax.hlines(y, *g["ci95"], color=col, lw=2)
                ax.scatter([g["mean"]], [y], color=col, s=18, zorder=3, edgecolor="white",
                           label=lab if i == 0 else None)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(cfgs)), [c[1] for c in cfgs[::-1]])
        ax.set_xlabel("Gap vs random deletion (> 0 = more faithful)")
        ax.set_title(f"Fill: {'training mean' if fill == 'zero' else fill}", fontsize=8, loc="left")
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7)
    save(fig, "fig_controlled_xai")


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(R / "figures"))
    ap.add_argument("--data", default=str(ROOT / "data" / "pneumoniamnist.npz"))
    a = ap.parse_args(argv)
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    y = np.asarray(load_npz(a.data)["test"][1]).ravel()
    ens = ensembles(y)
    fig_model_comparison()
    fig_paired_differences()
    fig_confusion(y, ens)
    fig_classwise(y, ens)
    fig_roc(y, ens)
    fig_deletion()
    fig_cfr()
    fig_controlled()
    print("figures written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
