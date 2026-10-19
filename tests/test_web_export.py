"""The browser export (BatchNorm folded, float32) gives the same logits as PyTorch."""

import json

import numpy as np
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
