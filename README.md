# Explainable AI for Medical Image Classification (PneumoniaMNIST)

## Overview
A reproducible, CPU-friendly pipeline for **binary chest X-ray classification (normal vs pneumonia)** on [PneumoniaMNIST](https://github.com/MedMNIST/MedMNIST) (MedMNIST v2), with explainability (Grad-CAM, Integrated Gradients, occlusion) and checks on explanation quality. The starting point is the official MedMNIST repository (Apache-2.0); see `REPOSITORY_ANALYSIS.md`.

## Project status (2026-10-18)
- **Implemented and tested** (39 tests pass on synthetic data): data validation and leakage audit, small CNN and ResNet-18, training with validation-only selection, test evaluation gated behind `--final`, class-wise metrics with bootstrap CIs and calibration, XAI methods and quality metrics, CLI and Gradio inference with input validation.
- **Not yet run on real data:** the dataset download from `zenodo.org` is blocked by the network policy of the cloud environment used so far. **No real performance results exist yet**, so none are reported.

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
```
CPU timing on 4 vCPUs: small CNN ≈ 5 s/epoch, ResNet-18 ≈ 48 s/epoch at 28 px (see `MODEL_DESIGN.md`).

## Evaluation protocol
Official MedMNIST splits (train 4,708 / val 524 / test 624, as documented upstream). Normalisation fitted on train. Model, epoch and decision threshold chosen on validation only. Test evaluated once for **all** runs, so the test set is not used to pick a winner. Metrics: AUROC (primary), AUPRC, balanced accuracy, sensitivity, specificity, per-class precision/F1, ECE, bootstrap 95% CIs; mean ± SD over 3 seeds. Details: `EXPERIMENT_PLAN.md`, `XAI_PLAN.md`.

## Results
Filled in automatically by `python -m scripts.make_report --update-readme` (run by `scripts.run_all --final`) from the JSON files in `reports/`. Nothing below is typed by hand.

<!-- results:start -->
None yet. Results will be added here only after the scripts have been run on the real data.
<!-- results:end -->

## Limitations
- No patient identifiers: the official val split was carved from the source training set by image, so train/val patient overlap is possible and validation scores may be optimistic.
- Single-source pediatric data at 28 px; no external validation is possible here; findings may not generalise.
- No lesion annotations: explanations cannot be checked for localisation accuracy, only for faithfulness, stability and border reliance. Heatmaps do not show clinically valid reasoning.
- External images go through centre-crop + resize; the exact MedMNIST resampling is not verified, so a small preprocessing mismatch is possible.

## Medical-use notice
This is a research and educational project. It is not a medical device and must not be used to diagnose, treat, or make decisions about individual patients.

## License and attribution
- Upstream code: MedMNIST, Apache-2.0 (not vendored; only metadata such as URLs/MD5s/split sizes reused).
- Data: PneumoniaMNIST, CC BY 4.0. Please cite: Yang et al., "MedMNIST v2", *Scientific Data* 10, 41 (2023), DOI 10.1038/s41597-022-01721-8; Yang, Shi, Ni, "MedMNIST Classification Decathlon", ISBI 2021; Kermany et al., *Cell* 172(5) (2018), DOI 10.1016/j.cell.2018.02.010.
- This repository's own code: MIT License (see `LICENSE`). The license covers the code only; the dataset keeps its CC BY 4.0 terms.
