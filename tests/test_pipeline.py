"""Integration: train -> evaluate -> explain -> infer on synthetic data."""

import json
import os

import numpy as np
import pytest
from PIL import Image

from medxai.evaluate import evaluate_checkpoint


def test_training_outputs(trained_run):
    rd = trained_run["run_dir"]
    for f in ("best.pt", "summary.json", "history.json", "val_scores.npy"):
        assert os.path.exists(os.path.join(rd, f)), f
    assert trained_run["n_train"] == 80 and trained_run["n_val"] == 30
    assert 0.0 <= trained_run["val_at_0.5"]["auroc"] <= 1.0
    s = json.load(open(os.path.join(rd, "summary.json")))
    assert s["env"]["torch"] and s["data_md5"]


def test_training_reproducible(synth_npz, tmp_path):
    from medxai.train import TrainConfig, train

    kw = dict(data_path=synth_npz, model="small_cnn", model_kwargs={"width": 8}, epochs=2,
              batch_size=32, out_dir=str(tmp_path), seed=5)
    a = train(TrainConfig(run_name="a", **kw), verbose=False)
    b = train(TrainConfig(run_name="b", **kw), verbose=False)
    va = np.load(os.path.join(a["run_dir"], "val_scores.npy"))
    vb = np.load(os.path.join(b["run_dir"], "val_scores.npy"))
    assert np.allclose(va, vb, atol=1e-5)


def test_test_split_is_gated(trained_run, synth_npz):
    ckpt = os.path.join(trained_run["run_dir"], "best.pt")
    with pytest.raises(PermissionError):
        evaluate_checkpoint(ckpt, synth_npz, "test")
    r = evaluate_checkpoint(ckpt, synth_npz, "test", final=True, n_boot=50)
    assert r["n"] == 30 and "ci95_at_0.5" in r
    assert os.path.exists(os.path.join(trained_run["run_dir"], "test_access.log"))


def test_input_size_mismatch(trained_run, tmp_path):
    from medxai.data import make_synthetic_npz

    p = make_synthetic_npz(str(tmp_path / "s64.npz"), size=64, counts=(4, 4, 4))
    with pytest.raises(ValueError, match="expects 28"):
        evaluate_checkpoint(os.path.join(trained_run["run_dir"], "best.pt"), p, "val", save=False)


def test_xai_methods(trained_run, synth_npz):
    import torch

    from medxai.data import load_npz
    from medxai.models import load_checkpoint
    from medxai.xai import (border_mass, deletion_auc, explain, overlay, perturbation_stability,
                            random_deletion_auc, rank_agreement, topk_iou)

    model, meta = load_checkpoint(os.path.join(trained_run["run_dir"], "best.pt"))
    imgs, _ = load_npz(synth_npz)["val"]
    n = meta["normalizer"]
    x = torch.from_numpy((imgs[:1].astype(np.float32) / 255 - n["mean"]) / n["std"]).unsqueeze(1)
    for m in ("gradcam", "ig", "occlusion"):
        h = explain(model, x, m)
        assert h.shape == (28, 28) and np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1
        assert 0 <= border_mass(h) <= 1 or np.isnan(border_mass(h))
        assert 0 <= deletion_auc(model, x, h) <= 1
    assert 0 <= random_deletion_auc(model, x, n=2) <= 1
    st = perturbation_stability(model, x, "gradcam", n=2)
    assert np.isnan(st) or -1 <= st <= 1  # NaN when Grad-CAM is all-zero (undefined correlation)
    assert np.isnan(rank_agreement(np.zeros((4, 4)), np.ones((4, 4))))
    h = explain(model, x, "ig")
    assert rank_agreement(h, h) == pytest.approx(1.0)
    assert topk_iou(h, h) == 1.0
    assert overlay(imgs[0] / 255, h).shape == (28, 28, 3)
    with pytest.raises(ValueError):
        explain(model, x, "magic")


def test_border_mass_reference():
    from medxai.xai import border_mass

    assert border_mass(np.ones((28, 28)), 0.1) == pytest.approx(1 - (22 * 22) / 28 ** 2)  # 3 px border
    edge = np.zeros((28, 28))
    edge[0, :] = 1
    assert border_mass(edge) == 1.0


def test_inference_valid_and_invalid(trained_run, tmp_path):
    from medxai.infer import InputError, Predictor

    pred = Predictor(os.path.join(trained_run["run_dir"], "best.pt"))
    g = tmp_path / "x.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 240), dtype=np.uint8)).save(g)
    out = pred.predict(str(g), explain_method="gradcam")
    assert abs(sum(out["scores"].values()) - 1) < 1e-5
    assert "RESEARCH USE ONLY" in out["notice"] and out["_heatmap"].shape == (28, 28)

    color = tmp_path / "c.png"
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[..., 0] = 255
    Image.fromarray(arr).save(color)
    tiny = tmp_path / "t.png"
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(tiny)
    txt = tmp_path / "a.txt"
    txt.write_text("hi")
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"\x89PNG not really")
    empty = tmp_path / "e.png"
    empty.write_bytes(b"")
    for bad in (color, tiny, txt, corrupt, empty, tmp_path / "missing.png"):
        with pytest.raises(InputError):
            pred.predict(str(bad))
    with pytest.raises(FileNotFoundError):
        Predictor(str(tmp_path / "no.pt"))


def test_rgb_grayscale_accepted(trained_run, tmp_path):
    from medxai.infer import Predictor

    p = tmp_path / "g.jpg"
    gray = np.random.default_rng(1).integers(0, 255, (100, 100), dtype=np.uint8)
    Image.fromarray(np.stack([gray] * 3, -1)).save(p)
    assert "scores" in Predictor(os.path.join(trained_run["run_dir"], "best.pt")).predict(str(p))


def test_cli_scripts(trained_run, synth_npz, tmp_path):
    from scripts import evaluate, explain, predict, validate_data

    assert validate_data.main(["--data", synth_npz, "--no-official-counts",
                               "--out", str(tmp_path / "v.json")]) == 0
    assert validate_data.main(["--data", synth_npz]) == 1  # counts differ from official
    ckpt = os.path.join(trained_run["run_dir"], "best.pt")
    assert evaluate.main(["--ckpt", ckpt, "--data", synth_npz, "--split", "test"]) == 2
    assert evaluate.main(["--ckpt", ckpt, "--data", synth_npz, "--split", "val", "--n-boot", "20"]) == 0
    out = tmp_path / "xai"
    assert explain.main(["--ckpts", ckpt, ckpt, "--data", synth_npz, "--n-per-group", "1",
                         "--n-stats", "4", "--out", str(out)]) == 0
    assert (out / "examples.png").exists() and (out / "xai_stats.json").exists()
    assert explain.main(["--ckpts", ckpt, "--data", synth_npz, "--split", "test", "--out", str(out)]) == 2
    img = tmp_path / "i.png"
    Image.fromarray(np.full((50, 50), 120, np.uint8)).save(img)
    assert predict.main(["--ckpt", ckpt, "--image", str(img), "--explain", "gradcam",
                         "--out", str(tmp_path / "h.png")]) == 0
    assert predict.main(["--ckpt", ckpt, "--image", str(tmp_path / "zz.png")]) == 2


def test_experiment_matrix_and_final(synth_npz, tmp_path):
    import yaml

    from scripts import final_eval, run_experiments

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({"model": "small_cnn", "model_kwargs": {"width": 8}, "epochs": 2,
                                   "batch_size": 32}))
    rep = tmp_path / "val.json"
    assert run_experiments.main(["--configs", str(cfg), "--seeds", "0", "1", "--data-path", synth_npz,
                                 "--out-dir", str(tmp_path / "runs"), "--report", str(rep)]) == 0
    v = json.load(open(rep))
    assert v["_selected"] == "tiny" and len(v["tiny"]["val_at_0.5"]["auroc"]["values"]) == 2
    assert final_eval.main(["--val-report", str(rep), "--data", synth_npz]) == 2
    assert final_eval.main(["--val-report", str(rep), "--data", synth_npz, "--final", "--n-boot", "20",
                            "--out", str(tmp_path / "t.json")]) == 0


def test_unknown_config_key(tmp_path):
    from medxai.train import TrainConfig

    p = tmp_path / "c.yaml"
    p.write_text("modle: small_cnn\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_file(str(p))


def test_optimizer_scheduler_options(synth_npz, tmp_path):
    from medxai.train import TrainConfig, train

    s = train(TrainConfig(data_path=synth_npz, model_kwargs={"width": 8}, epochs=2, batch_size=32,
                          optimizer="adam", scheduler="multistep", milestones=[1], out_dir=str(tmp_path),
                          run_name="ms"), verbose=False)
    assert s["best_epoch"] >= 1
    with pytest.raises(ValueError):
        train(TrainConfig(data_path=synth_npz, epochs=1, optimizer="sgdx", out_dir=str(tmp_path)), verbose=False)


def test_all_repo_configs_parse():
    import glob

    from medxai.train import TrainConfig

    paths = glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.yaml"))
    assert paths
    for p in paths:
        TrainConfig.from_file(p)


def test_run_all_and_report(synth_npz, tmp_path):
    import yaml

    from scripts import make_report, run_all

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({"model": "small_cnn", "model_kwargs": {"width": 8}, "epochs": 2,
                                   "batch_size": 32}))
    reports = tmp_path / "reports"
    args = ["--data", synth_npz, "--synthetic", "--quick", "--configs", str(cfg),
            "--runs-dir", str(tmp_path / "runs"), "--reports-dir", str(reports)]
    assert run_all.main(args) == 0
    assert not (reports / "test_summary.json").exists()  # test split untouched without --final
    text = (reports / "RESULTS.md").read_text()
    assert "E7 Held-out test" in text and "Not run yet. The test split" in text
    assert "| tiny | 1 |" in text and (reports / "xai_tiny_val" / "xai_stats.json").exists()

    assert run_all.main(args + ["--final"]) == 0
    assert (reports / "test_summary.json").exists()
    text = (reports / "RESULTS.md").read_text()
    assert "threshold chosen on val" in text and "Selected by validation" in text

    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{make_report.START}\nPLACEHOLDER\n{make_report.END}\nafter\n")
    assert make_report.main(["--reports", str(reports), "--out", str(tmp_path / "r.md"),
                             "--update-readme", "--readme", str(readme)]) == 0
    r = readme.read_text()
    assert "PLACEHOLDER" not in r and "E0 Data validation" in r and r.endswith("after\n")
    readme.write_text("no markers\n")
    with pytest.raises(ValueError):
        make_report.main(["--reports", str(reports), "--update-readme", "--readme", str(readme)])


def test_report_with_no_results(tmp_path):
    from scripts import make_report

    assert make_report.main(["--reports", str(tmp_path / "empty"), "--out", str(tmp_path / "r.md")]) == 0
    assert (tmp_path / "r.md").read_text().count("Not run yet") == 5


def test_run_all_larger_size(tmp_path):
    """--size picks the matching data file name and keeps outputs separate from the 28 px ones."""
    import yaml

    from medxai.data import make_synthetic_npz
    from scripts import run_all

    data = tmp_path / "data"
    data.mkdir()
    make_synthetic_npz(str(data / "pneumoniamnist_64.npz"), size=64, counts=(40, 16, 16))
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({"model": "small_cnn", "model_kwargs": {"width": 8}, "epochs": 1,
                                   "batch_size": 32}))
    reports = tmp_path / "reports"
    os.chdir(tmp_path)  # default data path data/pneumoniamnist_64.npz is relative
    try:
        assert run_all.main(["--size", "64", "--synthetic", "--quick", "--configs", str(cfg),
                             "--runs-dir", str(tmp_path / "runs"), "--reports-dir", str(reports)]) == 0
    finally:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    v = json.load(open(reports / "data_validation_64.json"))
    assert v["file"] == "pneumoniamnist_64.npz"
    assert "pneumoniamnist_64.npz" in (reports / "RESULTS.md").read_text()
