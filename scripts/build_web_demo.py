"""Build the self-contained in-browser demo page from an exported model.

Fills web/demo_template.html with the model JSON (scripts/export_web_model.py),
web/engine.js and the test numbers of the same configuration from
reports/test_summary.json. Every number on the page comes from those files.

Usage: python -m scripts.build_web_demo --model web_model.json [--out web/pneumonia-demo.html]
       [--reports reports] [--config small_cnn] [--run-id local]
"""

import argparse
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


def _ms(d):
    return f"{d['mean']:.3f} ± {d['std']:.3f}"


def build(model: dict, test_summary: dict | None, config: str, run_id: str) -> str:
    v = model["run"]["val_at_0.5"]
    if test_summary and config in test_summary["models"]:
        t = test_summary["models"][config]["aggregate"]["at_0.5"]
        test = [_ms(t["auroc"]), _ms(t["pneumonia_recall(sensitivity)"]), _ms(t["normal_recall(specificity)"])]
    else:
        test = ["not run yet"] * 3
    rep = {
        "THR": f"{model['val_threshold_youden']:.2f}", "NPARAMS": f"{model['run']['n_params']:,}",
        "V_AUROC": f"{v['auroc']:.3f}", "V_SENS": f"{v['per_class']['pneumonia']['recall(sensitivity)']:.3f}",
        "V_SPEC": f"{v['per_class']['normal']['recall(specificity)']:.3f}",
        "T_AUROC": test[0], "T_SENS": test[1], "T_SPEC": test[2], "RUN_ID": run_id,
        "MODEL_JSON": json.dumps(model, separators=(",", ":"), allow_nan=False),
    }
    with open(os.path.join(HERE, "engine.js"), encoding="utf-8") as f:
        rep["ENGINE_JS"] = f.read()
    with open(os.path.join(HERE, "demo_template.html"), encoding="utf-8") as f:
        html = f.read()
    for k, val in rep.items():
        html = html.replace("{{" + k + "}}", val)
    if "{{" in html:
        raise ValueError("unfilled placeholder: " + html[html.index("{{"):html.index("{{") + 30])
    return html


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="web/pneumonia-demo.html")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--config", default="small_cnn")
    ap.add_argument("--run-id", default="local")
    a = ap.parse_args(argv)
    with open(a.model) as f:
        model = json.load(f)
    ts = os.path.join(a.reports, "test_summary.json")
    test = json.load(open(ts)) if os.path.exists(ts) else None
    html = build(model, test, a.config, a.run_id)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
