"""E9 CAM-faithfulness regularisation: CAM exactness, loss behaviour, training and comparison."""

import json
import os

import numpy as np
import pytest
import torch

from medxai.data import augment_batch
from medxai.faithful import cfr_penalty, deletion_loss, equivariance_loss, frozen_bn_stats, top_fraction_mask
from medxai.models import SmallCNN
from medxai.xai import grad_cam


def _model():
    torch.manual_seed(0)
    return SmallCNN(width=8)


def test_cam_matches_logit_and_gradcam():
    m = _model().eval()
    x = torch.randn(3, 1, 28, 28)
    feats = m.features(x)
    y = torch.tensor([0, 1, 1])
    cam = m.cam(feats, y)
    logits = m(x)
    # logit = mean(CAM) + bias (GAP + linear head; dropout is off in eval)
    assert torch.allclose(cam.mean((1, 2)) + m.head[-1].bias[y], logits.gather(1, y[:, None])[:, 0], atol=1e-5)
    # Grad-CAM uses ReLU(sum_k mean(grad) * A_k); with a GAP head mean(grad)=w/(HW), so maps agree
    up = torch.nn.functional.interpolate(torch.relu(cam[:1, None]), size=(28, 28), mode="bilinear",
                                         align_corners=False)[0, 0].detach().numpy()
    gc = grad_cam(m, x[:1], target=0)
    if up.max() > up.min():
        up = (up - up.min()) / (up.max() - up.min())
        assert np.allclose(up, gc, atol=1e-4)


def test_augment_theta_does_not_change_draws():
    x = torch.randn(4, 1, 28, 28)
    a = augment_batch(x, torch.Generator().manual_seed(1))
    b, theta = augment_batch(x, torch.Generator().manual_seed(1), return_theta=True)
    assert torch.equal(a, b) and theta.shape == (4, 2, 3)


def test_top_fraction_mask():
    h = torch.arange(100.0).view(1, 1, 10, 10)
    m = top_fraction_mask(h, 0.2)
    assert m.sum() == 20 and m.view(-1)[-20:].sum() == 20


def test_losses_finite_and_bn_stats_untouched():
    m = _model().train()
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 2, (8,))
    xa, theta = augment_batch(x, torch.Generator().manual_seed(0), return_theta=True)
    before = [b.clone() for b in m.buffers()]
    feats = m.features(xa)  # the main forward may update BN stats; snapshot after it
    after_main = [b.clone() for b in m.buffers()]
    d = deletion_loss(m, xa, feats, y)
    e = equivariance_loss(m, x, theta, feats, y)
    assert torch.isfinite(d) and torch.isfinite(e) and 0 <= d.item() <= 0.5 and e.item() >= 0
    assert all(torch.equal(a, b) for a, b in zip(after_main, m.buffers()))
    assert not all(torch.equal(a, b) for a, b in zip(before, after_main))
    (d + e).backward()
    assert m.block1[0].weight.grad is not None and torch.isfinite(m.block1[0].weight.grad).all()


def test_equivariance_zero_for_identity():
    m = _model().eval()
    x, y = torch.randn(2, 1, 28, 28), torch.tensor([0, 1])
    theta = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]]).repeat(2, 1, 1)
    assert equivariance_loss(m, x, theta, m.features(x), y).item() < 1e-6


def test_penalty_zero_weights():
    m = _model()
    x, y = torch.randn(2, 1, 28, 28), torch.tensor([0, 1])
    pen, parts = cfr_penalty(m, x, m.features(x), y, x, None, 0.0, 0.0, 0.2)
    assert float(pen) == 0 and parts == {}


def test_frozen_bn_stats_restores():
    m = _model().train()
    snap = [b.clone() for b in m.buffers()]
    with frozen_bn_stats(m):
        m(torch.randn(4, 1, 28, 28))
    assert all(torch.equal(a, b) for a, b in zip(snap, m.buffers()))


def test_cfr_training_and_compare(synth_npz, tmp_path):
    from medxai.train import TrainConfig, train
    from scripts import compare_xai

    kw = dict(data_path=synth_npz, model_kwargs={"width": 8}, epochs=2, batch_size=32, out_dir=str(tmp_path))
    base = train(TrainConfig(run_name="base", **kw), verbose=False)
    cfr = train(TrainConfig(run_name="cfr", cam_del_weight=1.0, cam_eq_weight=0.5, **kw), verbose=False)
    hist = json.load(open(os.path.join(cfr["run_dir"], "history.json")))
    assert "train_del_loss" in hist[0] and "train_eq_loss" in hist[0]
    with pytest.raises(ValueError):
        train(TrainConfig(data_path=synth_npz, model="resnet18", epochs=1, cam_del_weight=1.0,
                          out_dir=str(tmp_path)), verbose=False)
    out = tmp_path / "cmp.json"
    b, c = os.path.join(base["run_dir"], "best.pt"), os.path.join(cfr["run_dir"], "best.pt")
    args = ["--group", f"baseline={b}", "--group", f"cfr={c}", "--ref", "baseline", "--data", synth_npz,
            "--n-images", "6", "--n-boot", "50", "--out", str(out)]
    assert compare_xai.main(args) == 0
    r = json.load(open(out))
    assert r["n_images"] == 6 and "gradcam_gap" in r["groups"]["cfr"]["vs_ref"]
    assert compare_xai.main(args[:-2] + ["--split", "test", "--out", str(out)]) == 2
