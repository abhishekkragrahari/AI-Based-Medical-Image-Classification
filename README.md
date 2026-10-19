# Explainable AI for Medical Image Classification (PneumoniaMNIST)

## Overview
A reproducible, CPU-friendly pipeline for **binary chest X-ray classification (normal vs pneumonia)** on [PneumoniaMNIST](https://github.com/MedMNIST/MedMNIST) (MedMNIST v2), with explainability (Grad-CAM, Integrated Gradients, occlusion) and checks on explanation quality. The starting point is the official MedMNIST repository (Apache-2.0); see `REPOSITORY_ANALYSIS.md`.

## Project status (2026-10-18)
- **Implemented and tested** (39 tests pass on synthetic data): data validation and leakage audit, small CNN and ResNet-18, training with validation-only selection, test evaluation gated behind `--final`, class-wise metrics with bootstrap CIs and calibration, XAI methods and quality metrics, CLI and Gradio inference with input validation.
- **Run on real data (2026-10-18):** E0, E2–E4, E5 and the one-time E7 test evaluation were run by the `experiments` GitHub Actions workflow (run 39142842061, CPU runner, 3 seeds). Results below are generated from `reports/*.json`. E1 (MedMNIST ResNet-18 protocol) has not been run yet.

## Goals
- Reproduce a clearly defined baseline (MedMNIST v2 ResNet-18 protocol, E1).
- Compare a small CNN and ResNet-18 under a fixed evaluation protocol (E2–E4).
- Evaluate explanation faithfulness, stability and shortcut reliance (E5).
- Report class-wise performance, limitations and failure cases.

## Layout
```
medxai/        data.py (loading, validation, normalisation, augmentation) · models.py · train.py
               metrics.py · evaluate.py (gated test) · xai.py · infer.py
scripts/       download_data · validate_data · train · run_experiments · evaluate · final_eval · explain · predict
               run_all (whole pipeline) · make_report (reports/*.json -> Markdown)
configs/       small_cnn, resnet18, resnet18_medmnist_protocol, ablations
tests/         pytest suite (synthetic data only); CI: .github/workflows/tests.yml
app.py         local Gradio demo
web/           in-browser demo (engine.js mirrors infer.py + occlusion; demo_template.html)
reports/       small JSON/PNG outputs (committed); runs/ and data/ are git-ignored
```

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate        # Python 3.13 tested
pip install -r requirements.txt
python -m pytest -q                                  # synthetic-data tests

# One command for everything (each step below, in order):
python -m scripts.run_all --quick                    # ~1-2 min smoke run -> runs/quick, reports/quick
python -m scripts.run_all                            # E0, E2-E4, E5 on val (~45 min CPU); test untouched
python -m scripts.run_all --final                    # ...then E7 once and fill the README Results
                                                     # add --include-e1 for the ~4 h ResNet-18 protocol
# No local download access? Run the 'experiments' workflow in GitHub Actions; it pushes results to a results/<run id> branch.

# Or step by step:

python -m scripts.download_data --size 28            # needs access to zenodo.org; verifies MD5
python -m scripts.validate_data --out reports/data_validation_28.json            # E0
python -m scripts.run_experiments --seeds 0 1 2 \
    --configs configs/small_cnn.yaml configs/resnet18.yaml \
              configs/small_cnn_classweight.yaml configs/small_cnn_noaug.yaml \
    --report reports/val_summary.json                                            # E2-E4 (val only)
python -m scripts.run_experiments --seeds 0 1 2 --configs configs/resnet18_medmnist_protocol.yaml \
    --report reports/val_e1.json                                                 # E1
python -m scripts.explain --ckpts runs/small_cnn_s0/best.pt runs/small_cnn_s1/best.pt runs/small_cnn_s2/best.pt \
    --split val --out reports/xai_small_cnn_val                                  # E5
python -m scripts.final_eval --val-report reports/val_summary.json --final       # E7, once

python -m scripts.predict --ckpt runs/small_cnn_s0/best.pt --image some_xray.png --explain gradcam --out heat.png
python app.py --ckpt runs/small_cnn_s0/best.pt                                   # http://127.0.0.1:7860

# Browser demo (no server): export a small CNN run and build one self-contained HTML page
python -m scripts.export_web_model --run runs/small_cnn_s0 --data data/pneumoniamnist.npz --out web_model.json
python -m scripts.build_web_demo --model web_model.json --out web/pneumonia-demo.html   # open it in a browser
# The 'web-demo-model' workflow does both on GitHub Actions.
```
CPU timing on 4 vCPUs: small CNN ≈ 5 s/epoch, ResNet-18 ≈ 48 s/epoch at 28 px (see `MODEL_DESIGN.md`).

## Evaluation protocol
Official MedMNIST splits (train 4,708 / val 524 / test 624, as documented upstream). Normalisation fitted on train. Model, epoch and decision threshold chosen on validation only. Test evaluated once for **all** runs, so the test set is not used to pick a winner. Metrics: AUROC (primary), AUPRC, balanced accuracy, sensitivity, specificity, per-class precision/F1, ECE, bootstrap 95% CIs; mean ± SD over 3 seeds. Details: `EXPERIMENT_PLAN.md`, `XAI_PLAN.md`.

## Results
Filled in automatically by `python -m scripts.make_report --update-readme` (run by `scripts.run_all --final`) from the JSON files in `reports/`. Nothing below is typed by hand.

<!-- results:start -->
_Generated by `python -m scripts.make_report` from `reports/`. Research outputs only, not clinical evidence._

### E0 Data validation
File `pneumoniamnist.npz`, MD5 matches official: True, checks passed: True.

| Split | Images | Normal (0) | Pneumonia (1) | Exact dups within split |
|---|---|---|---|---|
| train | 4708 | 1214 | 3494 | 18 |
| val | 524 | 135 | 389 | 0 |
| test | 624 | 234 | 390 | 3 |

Exact duplicates across splits: train-val 8, train-test 0, val-test 0.

### E2–E4 Model comparison (validation split, threshold 0.5, mean ± SD over seeds)
| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.978 ± 0.003 | 0.991 ± 0.004 | 0.965 ± 0.004 | 0.979 ± 0.004 | 0.013 ± 0.003 |
| resnet18 | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.977 ± 0.007 | 0.976 ± 0.006 | 0.978 ± 0.020 | 0.970 ± 0.002 | 0.017 ± 0.000 |
| small_cnn_classweight | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.977 ± 0.002 | 0.969 ± 0.019 | 0.985 ± 0.015 | 0.966 ± 0.013 | 0.015 ± 0.000 |
| small_cnn_noaug | 3 | 0.997 ± 0.001 | 0.999 ± 0.000 | 0.964 ± 0.002 | 0.967 ± 0.032 | 0.960 ± 0.035 | 0.955 ± 0.017 | 0.015 ± 0.006 |

Validation without images that also appear in train (516 images):

| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.978 ± 0.003 | 0.990 ± 0.004 | 0.965 ± 0.004 | 0.979 ± 0.004 | 0.013 ± 0.003 |
| resnet18 | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.977 ± 0.007 | 0.976 ± 0.006 | 0.978 ± 0.020 | 0.970 ± 0.002 | 0.017 ± 0.001 |
| small_cnn_classweight | 3 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.977 ± 0.003 | 0.969 ± 0.020 | 0.985 ± 0.015 | 0.966 ± 0.013 | 0.016 ± 0.001 |
| small_cnn_noaug | 3 | 0.997 ± 0.001 | 0.999 ± 0.000 | 0.963 ± 0.002 | 0.966 ± 0.032 | 0.960 ± 0.035 | 0.955 ± 0.017 | 0.016 ± 0.006 |

Selected by validation (highest mean val AUROC across seeds): **resnet18**.

### E1 MedMNIST ResNet-18 protocol (validation split, threshold 0.5, mean ± SD over seeds)
Not run yet.

### E5 Explanation quality (validation split)

`xai_resnet18_val` (val, 100 samples; random-deletion AUC 0.698 ± 0.252, uniform border reference 0.383):

| Method | Deletion AUC (lower = more faithful) | Border mass | Stability (Spearman) | Cross-seed top-10% IoU |
|---|---|---|---|---|
| gradcam | 0.751 ± 0.261 | 0.279 ± 0.064 | 0.997 ± 0.004 | 0.428 ± 0.244 |
| ig | 0.695 ± 0.290 | 0.313 ± 0.069 | 0.902 ± 0.037 | 0.246 ± 0.074 |
| occlusion | 0.555 ± 0.315 | 0.138 ± 0.114 | 0.984 ± 0.012 | 0.445 ± 0.203 |

Examples: `reports/xai_resnet18_val/examples.png`.

### E7 Held-out test (evaluated once, all models reported)

Test, threshold 0.5:

| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | 3 | 0.979 ± 0.005 | 0.984 ± 0.006 | 0.875 ± 0.010 | 0.998 ± 0.003 | 0.751 ± 0.024 | 0.893 ± 0.009 | 0.066 ± 0.005 |
| resnet18 | 3 | 0.980 ± 0.005 | 0.987 ± 0.004 | 0.872 ± 0.038 | 0.991 ± 0.001 | 0.752 ± 0.078 | 0.888 ± 0.034 | 0.067 ± 0.019 |
| small_cnn_classweight | 3 | 0.979 ± 0.003 | 0.986 ± 0.003 | 0.897 ± 0.011 | 0.985 ± 0.013 | 0.809 ± 0.035 | 0.911 ± 0.008 | 0.048 ± 0.016 |
| small_cnn_noaug | 3 | 0.965 ± 0.013 | 0.973 ± 0.014 | 0.849 ± 0.055 | 0.991 ± 0.010 | 0.708 ± 0.121 | 0.867 ± 0.049 | 0.088 ± 0.041 |

Test, threshold chosen on val (Youden):

| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | 3 | 0.979 ± 0.005 | 0.984 ± 0.006 | 0.907 ± 0.010 | 0.990 ± 0.008 | 0.823 ± 0.027 | 0.920 ± 0.007 | 0.066 ± 0.005 |
| resnet18 | 3 | 0.980 ± 0.005 | 0.987 ± 0.004 | 0.874 ± 0.012 | 0.990 ± 0.003 | 0.758 ± 0.024 | 0.890 ± 0.011 | 0.067 ± 0.019 |
| small_cnn_classweight | 3 | 0.979 ± 0.003 | 0.986 ± 0.003 | 0.889 ± 0.020 | 0.987 ± 0.004 | 0.791 ± 0.035 | 0.903 ± 0.018 | 0.048 ± 0.016 |
| small_cnn_noaug | 3 | 0.965 ± 0.013 | 0.973 ± 0.014 | 0.857 ± 0.039 | 0.994 ± 0.004 | 0.719 ± 0.082 | 0.875 ± 0.035 | 0.088 ± 0.041 |

Test, threshold chosen on de-duplicated val (Youden):

| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | 3 | 0.979 ± 0.005 | 0.984 ± 0.006 | 0.907 ± 0.010 | 0.990 ± 0.008 | 0.823 ± 0.027 | 0.920 ± 0.007 | 0.066 ± 0.005 |
| resnet18 | 3 | 0.980 ± 0.005 | 0.987 ± 0.004 | 0.874 ± 0.012 | 0.990 ± 0.003 | 0.758 ± 0.024 | 0.890 ± 0.011 | 0.067 ± 0.019 |
| small_cnn_classweight | 3 | 0.979 ± 0.003 | 0.986 ± 0.003 | 0.889 ± 0.020 | 0.987 ± 0.004 | 0.791 ± 0.035 | 0.903 ± 0.018 | 0.048 ± 0.016 |
| small_cnn_noaug | 3 | 0.965 ± 0.013 | 0.973 ± 0.014 | 0.857 ± 0.039 | 0.994 ± 0.004 | 0.719 ± 0.082 | 0.875 ± 0.035 | 0.088 ± 0.041 |

Test, seed ensemble (mean score of the seeds), threshold 0.5 (± 0 = single value):

| Model | Seeds | AUROC | AUPRC | Bal. acc | Sensitivity | Specificity | Macro F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| small_cnn | ens. | 0.982 ± 0.000 | 0.988 ± 0.000 | 0.883 ± 0.000 | 0.997 ± 0.000 | 0.769 ± 0.000 | 0.901 ± 0.000 | 0.051 ± 0.000 |
| resnet18 | ens. | 0.982 ± 0.000 | 0.988 ± 0.000 | 0.880 ± 0.000 | 0.995 ± 0.000 | 0.765 ± 0.000 | 0.897 ± 0.000 | 0.052 ± 0.000 |
| small_cnn_classweight | ens. | 0.983 ± 0.000 | 0.989 ± 0.000 | 0.897 ± 0.000 | 0.987 ± 0.000 | 0.808 ± 0.000 | 0.911 ± 0.000 | 0.040 ± 0.000 |
| small_cnn_noaug | ens. | 0.976 ± 0.000 | 0.984 ± 0.000 | 0.854 ± 0.000 | 0.995 ± 0.000 | 0.714 ± 0.000 | 0.874 ± 0.000 | 0.057 ± 0.000 |

Paired bootstrap, test AUROC of each seed ensemble minus **resnet18**:
- small_cnn: -0.0002 (95% CI -0.0053 to +0.0045, bootstrap p ≈ 0.926)
- small_cnn_classweight: +0.0007 (95% CI -0.0024 to +0.0040, bootstrap p ≈ 0.712)
- small_cnn_noaug: -0.0059 (95% CI -0.0115 to -0.0008, bootstrap p ≈ 0.024)

Model selected on validation before the test run: **resnet18**.

Per-run test AUROC bootstrap 95% CIs (threshold-free):
- small_cnn: [0.974, 0.990], [0.960, 0.985], [0.971, 0.990]
- resnet18: [0.978, 0.992], [0.963, 0.985], [0.970, 0.989]
- small_cnn_classweight: [0.970, 0.988], [0.964, 0.986], [0.971, 0.989]
- small_cnn_noaug: [0.929, 0.969], [0.953, 0.979], [0.965, 0.986]
<!-- results:end -->

## What the results show
Interpretation of the numbers above (all from `reports/`; n = 3 seeds, so no significance test is claimed):
- **Validation is optimistic.** Validation AUROC is ≈0.998 for every model, but test AUROC is 0.965–0.980. The gap is largest for specificity: at threshold 0.5 most models classify only ≈75% of normal test images as normal (e.g. ResNet-18 seed 1: 70 of 234 normal test images called pneumonia), while sensitivity stays ≥0.98. This fits the known limitation that val was split from the source training set by image, not by patient; E0 found 8 exact duplicate images shared between train and val (none with test).
- **No clear model winner.** ResNet-18 was selected by the pre-registered rule (highest mean val AUROC), but its margin over the small CNN is 0.0002, within seed variation, and test AUROC is also within overlapping ranges (0.980 ± 0.005 vs 0.979 ± 0.005). The small CNN has ~39× fewer parameters (287 k vs 11.2 M) and trained ~10× faster on CPU.
- **Ablations.** Removing augmentation lowered test AUROC (0.965 ± 0.013) and increased variance. Class weighting gave the highest test specificity at 0.5 (0.809 ± 0.035); this is an observation on the test set and was not used for selection.
- **Calibration and threshold.** Test ECE (≈0.05–0.09) is several times the val ECE (≈0.015). The val-chosen Youden threshold helped the small CNN (balanced accuracy 0.875 → 0.907) but not ResNet-18.
- **Explanations (ResNet-18, val, 100 images).** Occlusion has a lower deletion AUC than random deletion (0.555 vs 0.698), i.e. it ranks pixels the model relies on; Grad-CAM (0.751) and Integrated Gradients (0.695) do not beat random deletion at 28 px. Border mass is below the uniform reference (0.383) for all methods, so there is no sign of reliance on image borders. Agreement between seeds is low (top-10% IoU 0.25–0.45), so a single heatmap should not be read as *the* model's reasoning.

## Limitations
- No patient identifiers: the official val split was carved from the source training set by image, so train/val patient overlap is possible. The observed val→test drop (above) and 8 exact train/val duplicates are consistent with this; validation scores are optimistic.
- Single-source pediatric data at 28 px; no external validation is possible here; findings may not generalise.
- No lesion annotations: explanations cannot be checked for localisation accuracy, only for faithfulness, stability and border reliance. Heatmaps do not show clinically valid reasoning.
- External images go through centre-crop + resize; the exact MedMNIST resampling is not verified, so a small preprocessing mismatch is possible.

## Medical-use notice
This is a research and educational project. It is not a medical device and must not be used to diagnose, treat, or make decisions about individual patients.

## License and attribution
- Upstream code: MedMNIST, Apache-2.0 (not vendored; only metadata such as URLs/MD5s/split sizes reused).
- Data: PneumoniaMNIST, CC BY 4.0. Please cite: Yang et al., "MedMNIST v2", *Scientific Data* 10, 41 (2023), DOI 10.1038/s41597-022-01721-8; Yang, Shi, Ni, "MedMNIST Classification Decathlon", ISBI 2021; Kermany et al., *Cell* 172(5) (2018), DOI 10.1016/j.cell.2018.02.010.
- This repository's own code: MIT License (see `LICENSE`). The license covers the code only; the dataset keeps its CC BY 4.0 terms.
