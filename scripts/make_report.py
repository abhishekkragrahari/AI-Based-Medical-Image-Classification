"""Build a Markdown results report from the JSON files the pipeline writes.

Every number comes from reports/*.json; nothing is typed in by hand. Missing reports
are listed as "not run yet". With --update-readme the README section between the
results markers is replaced.

Usage: python -m scripts.make_report [--reports reports] [--out reports/RESULTS.md] [--update-readme]
"""

import argparse
import glob
import json
import os
import sys

START, END = "<!-- results:start -->", "<!-- results:end -->"
COLS = (("auroc", "AUROC"), ("auprc", "AUPRC"), ("balanced_accuracy", "Bal. acc"),
        ("pneumonia_recall(sensitivity)", "Sensitivity"), ("normal_recall(specificity)", "Specificity"),
        ("macro_f1", "Macro F1"), ("ece", "ECE"))


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ms(d):
    return f"{d['mean']:.3f} ± {d['std']:.3f}" if d and d.get("mean") is not None else "n/a"


def _metric_table(rows):
    """rows: list of (name, n_runs, metrics dict keyed like COLS)."""
    out = ["| Model | Seeds | " + " | ".join(c for _, c in COLS) + " |",
           "|---|---|" + "---|" * len(COLS)]
    for name, n, m in rows:
        out.append(f"| {name} | {n} | " + " | ".join(_ms(m.get(k)) for k, _ in COLS) + " |")
    return out


def data_section(rep):
    lines = ["### E0 Data validation"]
    if rep is None:
        return lines + ["Not run yet (`python -m scripts.validate_data --out reports/data_validation_28.json`)."]
    st = rep["stats"]
    lines += [f"File `{rep['file']}`, MD5 matches official: {rep['md5_matches_official']}, "
              f"checks passed: {rep['ok']}.", "",
              "| Split | Images | Normal (0) | Pneumonia (1) | Exact dups within split |",
              "|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        if s in st and "class_counts" in st[s]:
            c = st[s]["class_counts"]
            lines.append(f"| {s} | {st[s]['n']} | {c['0']} | {c['1']} | "
                         f"{st[s]['n_within_split_exact_duplicates']} |")
    cross = st.get("cross_split_exact_duplicates", {})
    lines += ["", "Exact duplicates across splits: " + ", ".join(f"{k} {v}" for k, v in cross.items()) + "."]
    if rep["errors"]:
        lines += ["", "Errors: " + "; ".join(rep["errors"])]
    return lines


def val_section(rep, title):
    lines = [f"### {title} (validation split, threshold 0.5, mean ± SD over seeds)"]
    if rep is None:
        return lines + ["Not run yet."]
    rows = [(k, len(v["runs"]), v["val_at_0.5"]) for k, v in rep.items() if not k.startswith("_")]
    lines += _metric_table(rows)
    if "_selected" in rep:
        lines += ["", f"Selected by validation ({rep['_selection_rule']}): **{rep['_selected']}**."]
    return lines


def test_section(rep):
    lines = ["### E7 Held-out test (evaluated once, all models reported)"]
    if rep is None:
        return lines + ["Not run yet. The test split is only read by "
                        "`python -m scripts.final_eval --final`."]
    for at, label in (("at_0.5", "threshold 0.5"), ("at_val_youden", "threshold chosen on val (Youden)")):
        rows = [(k, len(v["runs"]), v["aggregate"][at]) for k, v in rep["models"].items()]
        lines += ["", f"Test, {label}:", ""] + _metric_table(rows)
    lines += ["", f"Model selected on validation before the test run: **{rep['selected_by_validation']}**."]
    lines += ["", "Per-run test AUROC bootstrap 95% CIs (threshold-free):"]
    for name, m in rep["models"].items():
        cis = [f"[{c['auroc'][0]:.3f}, {c['auroc'][1]:.3f}]" for c in m["per_run_ci95_at_0.5"] if "auroc" in c]
        lines.append(f"- {name}: " + (", ".join(cis) if cis else "n/a"))
    return lines


def xai_section(reports_dir):
    lines = ["### E5 Explanation quality (validation split)"]
    paths = sorted(glob.glob(os.path.join(reports_dir, "xai_*", "xai_stats.json")))
    if not paths:
        return lines + ["Not run yet."]
    for p in paths:
        s = _load(p)
        name = os.path.basename(os.path.dirname(p))
        lines += ["", f"`{name}` ({s['split']}, {s['n_samples_stats']} samples; random-deletion AUC "
                  f"{_ms(s['random_deletion_auc'])}, uniform border reference "
                  f"{s['uniform_border_mass_reference']:.3f}):", "",
                  "| Method | Deletion AUC (lower = more faithful) | Border mass | Stability (Spearman) | "
                  "Cross-seed top-10% IoU |", "|---|---|---|---|---|"]
        cs = s.get("cross_seed_top10pct_iou") or {}
        for m, d in s["per_method"].items():
            lines.append(f"| {m} | {_ms(d['deletion_auc'])} | {_ms(d['border_mass'])} | "
                         f"{_ms(d['stability'])} | {_ms(cs.get(m))} |")
        if os.path.exists(os.path.join(os.path.dirname(p), "examples.png")):
            lines.append(f"\nExamples: `{os.path.relpath(os.path.dirname(p))}/examples.png`.")
    return lines


def build(reports_dir):
    parts = [
        f"_Generated by `python -m scripts.make_report` from `{reports_dir}/`. "
        "Research outputs only, not clinical evidence._", "",
        *data_section(_load(os.path.join(reports_dir, "data_validation_28.json"))), "",
        *val_section(_load(os.path.join(reports_dir, "val_summary.json")), "E2–E4 Model comparison"), "",
        *val_section(_load(os.path.join(reports_dir, "val_e1.json")), "E1 MedMNIST ResNet-18 protocol"), "",
        *xai_section(reports_dir), "",
        *test_section(_load(os.path.join(reports_dir, "test_summary.json"))),
    ]
    return "\n".join(parts) + "\n"


def update_readme(path, body):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        raise ValueError(f"{path} has no {START} ... {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{head}{START}\n{body}{END}{tail}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None, help="default: <reports>/RESULTS.md")
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args(argv)
    body = build(a.reports)
    out = a.out or os.path.join(a.reports, "RESULTS.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {out}")
    if a.update_readme:
        update_readme(a.readme, body)
        print(f"updated {a.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
