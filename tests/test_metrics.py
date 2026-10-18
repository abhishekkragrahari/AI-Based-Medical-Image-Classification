import numpy as np
import pytest

from medxai.metrics import binary_metrics, bootstrap_ci, confusion, expected_calibration_error, youden_threshold


def test_confusion_and_classwise():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.7, 0.8, 0.4, 0.9])
    m = binary_metrics(y, s)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["per_class"]["pneumonia"]["recall(sensitivity)"] == pytest.approx(2 / 3)
    assert m["per_class"]["normal"]["recall(specificity)"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["per_class"]["pneumonia"]["support"] == 3


def test_perfect_and_auc():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == 1.0 and m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_single_class_auc_nan():
    m = binary_metrics(np.array([1, 1]), np.array([0.6, 0.7]))
    assert np.isnan(m["auroc"])


def test_errors():
    with pytest.raises(ValueError):
        binary_metrics(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError):
        binary_metrics(np.array([]), np.array([]))


def test_ece_calibrated_vs_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.where(p > 0.5, 0.99, 0.01)) > 0.15


def test_youden_and_bootstrap():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0.3, 0.1, 200), rng.normal(0.6, 0.1, 200)].clip(0, 1)
    t = youden_threshold(y, s)
    assert 0.35 < t < 0.55
    ci = bootstrap_ci(y, s, n_boot=200)
    m = binary_metrics(y, s)
    assert ci["auroc"][0] <= m["auroc"] <= ci["auroc"][1]
    assert confusion([1], [1])["tp"] == 1
