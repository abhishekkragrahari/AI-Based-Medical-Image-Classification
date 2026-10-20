"""The browser export (BatchNorm folded, float32) gives the same logits as PyTorch."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from medxai.data import load_npz
from medxai.models import load_checkpoint
from scripts.export_web_model import export, main, reference_forward


def test_export_matches_torch(trained_run, synth_npz, tmp_path):
    rd = trained_run["run_dir"]
    e = export(rd, synth_npz, n_samples=4)
    assert len(e["samples"]) == 4 and {s["label"] for s in e["samples"]} == {0, 1}
    assert all(s["split"] == "val" for s in e["samples"])
    model, meta = load_checkpoint(f"{rd}/best.pt")
    m, s = meta["normalizer"]["mean"], meta["normalizer"]["std"]
    images = load_npz(synth_npz)["val"][0][:5]
    for img in images:
        x = (img.astype(np.float32) / 255.0 - m) / s
        with torch.no_grad():
            ref = model(torch.from_numpy(x)[None, None].float()).numpy()[0]
        assert np.allclose(reference_forward(e, x), ref, atol=1e-4)

    out = tmp_path / "web.json"
    assert main(["--run", rd, "--out", str(out), "--n-samples", "0"]) == 0
    j = json.load(open(out))
    assert j["format"] == "medxai-web-1" and j["samples"] == [] and len(j["layers"]) == 6


def test_demo_page_builds(trained_run, synth_npz):
    from scripts.build_web_demo import build

    e = export(trained_run["run_dir"], synth_npz, n_samples=2)
    html = build(e, None, "small_cnn", "test")
    assert "{{" not in html and "Research use only" in html and '"format":"medxai-web-1"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_engine_matches_python(trained_run, synth_npz, tmp_path):
    """web/engine.js (the code the page runs) gives the same preprocessing, probabilities
    and occlusion heatmap as medxai.infer / medxai.xai."""
    from PIL import Image

    from medxai.infer import preprocess
    from medxai.xai import occlusion

    rd = trained_run["run_dir"]
    e = export(rd, None, n_samples=0)
    (tmp_path / "m.json").write_text(json.dumps(e))
    model, meta = load_checkpoint(f"{rd}/best.pt")
    g = np.asarray(Image.fromarray((np.random.default_rng(3).random((61, 47)) * 255).astype(np.uint8)))
    x = preprocess(g, 28, meta["normalizer"])
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0, 1].item()
    heat = occlusion(model, x)
    (tmp_path / "g.bin").write_bytes(g.tobytes())
    engine = os.path.join(os.path.dirname(__file__), "..", "web", "engine.js")
    js = f"""
    const E = require({json.dumps(os.path.abspath(engine))}), fs = require("fs");
    const m = E.load(JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'm.json'))}, "utf8")));
    const g = new Uint8Array(fs.readFileSync({json.dumps(str(tmp_path / 'g.bin'))}));
    const pre = E.preprocess(g, 47, 61, 28, m.meta.normalizer);
    E.occlusion(m, pre.x, 28).then(r => console.log(JSON.stringify({{
      p: E.softmax(E.forward(m, pre.x, 28))[1], heat: Array.from(r.heat), x: Array.from(pre.x)}})));
    """
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert np.allclose(out["x"], x.numpy().ravel(), atol=1e-5)
    assert abs(out["p"] - p) < 1e-4
    assert np.allclose(np.array(out["heat"]).reshape(28, 28), heat, atol=1e-3)
