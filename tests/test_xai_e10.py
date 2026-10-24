"""E10/M1 step 2: resolution-scaled occlusion, insertion AUC, fill strategies, patch units,
model randomisation, and multi-class/RGB evaluation, explanation and inference."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from medxai.data import load_npz, make_synthetic_npz, to_model_input
from medxai.models import build_model, load_checkpoint
from medxai.xai import (deletion_auc, gaussian_blur, insertion_auc, model_randomisation_test, occlusion,
                        overlay, random_deletion_auc, random_insertion_auc, randomisation_stages,
                        reference_image)


@pytest.fixture(scope="module")
def rgb_run(tmp_path_factory):
    from medxai.train import TrainConfig, train

    d = tmp_path_factory.mktemp("rgbrun")
    npz = make_synthetic_npz(str(d / "rgb.npz"), counts=(96, 32, 32), n_classes=4, channels=3)
    s = train(TrainConfig(data_path=npz, model="small_cnn", model_kwargs={"width": 8}, epochs=3,
                          batch_size=32, out_dir=str(d), run_name="mc"), verbose=False)
    return {"npz": npz, "run_dir": s["run_dir"]}


def _model_x(trained_run, synth_npz):
    model, meta = load_checkpoint(f"{trained_run['run_dir']}/best.pt")
    x = to_model_input(load_npz(synth_npz)["val"][0][:1], meta["normalizer"])
    return model, x


# ---- legacy behaviour is unchanged at the old defaults ---------------------------------
def _legacy_deletion_auc(model, x, heat, t, steps=20, fill=0.0):
    order = np.argsort(-np.ravel(heat))
    xs = []
    for s in range(steps + 1):
        xd = x.clone().view(1, x.shape[1], -1)
        k = int(round(order.size * s / steps))
        if k:
            xd[..., torch.from_numpy(order[:k])] = fill
        xs.append(xd.view_as(x))
    with torch.no_grad():
        probs = torch.softmax(model(torch.cat(xs)), 1)[:, t].numpy()
    return float(np.trapezoid(probs, dx=1.0 / steps))


def test_deletion_default_matches_legacy(trained_run, synth_npz, rng):
    model, x = _model_x(trained_run, synth_npz)
    heat = rng.random((28, 28))
    assert deletion_auc(model, x, heat, target=1) == pytest.approx(_legacy_deletion_auc(model, x, heat, 1),
                                                                   abs=1e-7)


def test_occlusion_default_matches_explicit_28px(trained_run, synth_npz):
    model, x = _model_x(trained_run, synth_npz)
    assert np.allclose(occlusion(model, x), occlusion(model, x, patch=4, stride=2), atol=1e-12)


# ---- new options -------------------------------------------------------------------------
def test_occlusion_scales_with_resolution():
    model = build_model("small_cnn", width=4).eval()
    for side in (28, 64):
        h = occlusion(model, torch.randn(1, 1, side, side))
        assert h.shape == (side, side) and 0 <= h.min() and h.max() <= 1


def test_reference_images():
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(reference_image(x, "zero"), torch.zeros_like(x))
    assert torch.equal(reference_image(x, 0.0), torch.zeros_like(x))
    b = reference_image(x, "blur")
    assert b.shape == x.shape and b.std() < x.std()  # smoothing reduces variance
    const = torch.full((1, 1, 16, 16), 3.0)
    assert torch.allclose(gaussian_blur(const, 2.0), const, atol=1e-5)  # blur preserves constants
    n1, n2 = reference_image(x, "noise", seed=1), reference_image(x, "noise", seed=1)
    assert torch.equal(n1, n2) and abs(float(n1.std()) - 1) < 0.1
    with pytest.raises(ValueError):
        reference_image(x, "median")


@pytest.mark.parametrize("fill", ["zero", "blur", "noise"])
@pytest.mark.parametrize("unit", ["pixel", "patch"])
def test_deletion_insertion_bounds(trained_run, synth_npz, rng, fill, unit):
    model, x = _model_x(trained_run, synth_npz)
    heat = rng.random((28, 28))
    for fn in (deletion_auc, insertion_auc):
        v = fn(model, x, heat, fill=fill, unit=unit)
        assert 0 <= v <= 1
    assert 0 <= random_insertion_auc(model, x, n=2, fill=fill, unit=unit) <= 1


def test_insertion_endpoints():
    """Insertion starts at the reference image and ends at x; deletion the reverse."""

    class Probe(torch.nn.Module):  # P(class 1) = sigmoid(mean pixel)
        def forward(self, x):
            m = x.mean(dim=(1, 2, 3))
            return torch.stack([torch.zeros_like(m), m], 1)

    x = torch.full((1, 1, 28, 28), 4.0)
    heat = np.random.default_rng(0).random((28, 28))
    # With a fraction f of pixels moved, deletion gives sigmoid(4(1-f)) and insertion sigmoid(4f):
    # mirror-image curves between p(ref)=0.5 and p(x)=sigmoid(4), so the two AUCs are equal.
    ins = insertion_auc(Probe(), x, heat, target=1)
    dele = deletion_auc(Probe(), x, heat, target=1)
    p_ref, p_x = 0.5, float(torch.sigmoid(torch.tensor(4.0)))
    assert p_ref < ins < p_x and p_ref < dele < p_x
    assert ins == pytest.approx(dele, abs=1e-6)
    # A blur reference of a constant image is the image itself: insertion is flat at p(x).
    assert insertion_auc(Probe(), x, heat, target=1, fill="blur") == pytest.approx(p_x, abs=1e-5)


def test_patch_unit_moves_whole_patches():
    """With a map that is constant inside each patch, pixel and patch rankings agree, so the
    curves match up to the step rounding."""
    model = build_model("small_cnn", width=4).eval()
    x = torch.randn(1, 1, 28, 28)
    blocks = np.kron(np.random.default_rng(1).random((14, 14)), np.ones((2, 2)))
    a = deletion_auc(model, x, blocks, target=0, unit="pixel")
    b = deletion_auc(model, x, blocks, target=0, unit="patch")
    assert a == pytest.approx(b, abs=0.05)
    with pytest.raises(ValueError):
        deletion_auc(model, x, blocks, unit="superpixel")


def test_randomisation_stages_order():
    small = [n for n, _ in randomisation_stages(build_model("small_cnn"))]
    assert small == ["head", "block3", "block2", "block1"]
    res = [n for n, _ in randomisation_stages(build_model("resnet18"))]
    assert res[0] == "net.fc" and res[-1] == "net.conv1" and "net.layer4" in res


def test_model_randomisation_test(trained_run, synth_npz):
    model, x = _model_x(trained_run, synth_npz)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = model_randomisation_test(model, x, "gradcam")
    assert [r["randomised_through"] for r in out] == ["head", "block3", "block2", "block1"]
    for r in out:
        assert np.isnan(r["spearman"]) or -1 <= r["spearman"] <= 1
        assert 0 <= r["top10pct_iou"] <= 1
    # The trained model is untouched.
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_overlay_rgb():
    o = overlay(np.random.default_rng(0).random((16, 16, 3)), np.random.default_rng(1).random((16, 16)))
    assert o.shape == (16, 16, 3) and 0 <= o.min() and o.max() <= 1


# ---- multi-class / RGB through the scripts ------------------------------------------------
def test_evaluate_multiclass(rgb_run):
    from medxai.evaluate import evaluate_checkpoint

    r = evaluate_checkpoint(f"{rgb_run['run_dir']}/best.pt", rgb_run["npz"], "val", n_boot=20)
    assert "at_argmax" in r and "at_0.5" not in r and r["at_argmax"]["n_classes"] == 4
    assert np.load(os.path.join(rgb_run["run_dir"], "val_scores.npy")).shape == (32, 4)
    with pytest.raises(PermissionError):
        evaluate_checkpoint(f"{rgb_run['run_dir']}/best.pt", rgb_run["npz"], "test")


def test_run_experiments_and_final_eval_multiclass(rgb_run, tmp_path):
    from scripts.final_eval import main as final_main
    from scripts.run_experiments import main as run_main

    cfg = tmp_path / "mc.yaml"
    cfg.write_text("model: small_cnn\nmodel_kwargs: {width: 8}\nepochs: 1\nbatch_size: 32\n")
    rep = tmp_path / "val.json"
    assert run_main(["--configs", str(cfg), "--seeds", "0", "1", "--data-path", rgb_run["npz"],
                     "--out-dir", str(tmp_path / "runs"), "--report", str(rep)]) == 0
    val = json.loads(rep.read_text())
    assert val["_selected"] == "mc" and "val_at_argmax" in val["mc"]
    out = tmp_path / "test.json"
    assert final_main(["--val-report", str(rep), "--data", rgb_run["npz"], "--out", str(out),
                       "--final", "--n-boot", "20"]) == 0
    t = json.loads(out.read_text())
    assert "at_argmax" in t["models"]["mc"]["aggregate"]
    assert t["models"]["mc"]["seed_ensemble_at_argmax"]["n_classes"] == 4


def test_explain_multiclass_rgb_with_new_checks(rgb_run, tmp_path):
    from scripts.explain import main as explain_main

    out = tmp_path / "xai"
    assert explain_main(["--ckpts", f"{rgb_run['run_dir']}/best.pt", "--data", rgb_run["npz"],
                         "--n-per-group", "1", "--n-stats", "3", "--out", str(out),
                         "--fills", "zero", "blur", "--units", "pixel", "patch",
                         "--randomisation", "1"]) == 0
    s = json.loads((out / "xai_stats.json").read_text())
    assert set(s["faithfulness"]) == {"zero/pixel", "zero/patch", "blur/pixel", "blur/patch"}
    g = s["faithfulness"]["blur/patch"]["occlusion"]
    assert g["deletion_gap"]["n"] + g["deletion_gap"]["n_undefined"] == 3
    assert set(s["model_randomisation"]["gradcam"]) == {"head", "block3", "block2", "block1"}
    # E5 keys are still written.
    d = s["per_method"]["ig"]["deletion_auc"]
    assert d["n"] + d["n_undefined"] == 3 and "random_deletion_auc" in s
    assert (out / "examples.png").exists()


def test_infer_rgb_multiclass(rgb_run, tmp_path):
    from medxai.infer import InputError, Predictor

    p = Predictor(f"{rgb_run['run_dir']}/best.pt")
    img = Image.fromarray(load_npz(rgb_run["npz"])["val"][0][0])
    out = p.predict(img, explain_method="gradcam")
    assert "predicted_label_argmax" in out and len(out["scores"]) == 4
    assert out["_input01"].shape == (28, 28, 3) and out["_heatmap"].shape == (28, 28)
    assert abs(sum(out["scores"].values()) - 1) < 1e-5
    with pytest.raises(InputError):
        p.predict(Image.fromarray(np.zeros((32, 32), np.int32)))  # 16/32-bit grayscale ("I" mode)


def test_infer_binary_grayscale_unchanged(trained_run):
    from medxai.infer import InputError, Predictor

    p = Predictor(f"{trained_run['run_dir']}/best.pt")
    out = p.predict(Image.fromarray((np.random.default_rng(0).random((40, 40)) * 255).astype(np.uint8)))
    assert set(out["scores"]) == {"normal", "pneumonia"} and "predicted_label_at_0.5" in out
    with pytest.raises(InputError, match="colour image"):
        rgb = np.zeros((32, 32, 3), np.uint8)
        rgb[..., 0] = 200
        p.predict(Image.fromarray(rgb))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_xai_on_cuda(trained_run, synth_npz, rng):
    model, x = _model_x(trained_run, synth_npz)
    model, xc = model.cuda(), x.cuda()
    heat = rng.random((28, 28))
    for m in ("gradcam", "ig", "occlusion"):
        from medxai.xai import explain

        assert explain(model, xc, m).shape == (28, 28)
    assert 0 <= deletion_auc(model, xc, heat, fill="blur", unit="patch") <= 1
    assert 0 <= random_deletion_auc(model, xc, n=2) <= 1
