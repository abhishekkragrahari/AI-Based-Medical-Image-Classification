# Experiment Plan

## Before experiments (frozen 2026-10-18, before any real data was seen)
- Research question: see RESEARCH_GAP.md.
- Dataset version: PneumoniaMNIST, MedMNIST v3.0.2 files, 28 px, MD5 `28209eda62fecd6e6a2d98b1501bb15f`.
- Split protocol: official train/val/test. Test only in E7 via `--final`.
- Primary metric: AUROC (val for selection; test with bootstrap 95% CI for reporting).
- Secondary: balanced accuracy, sensitivity, specificity, per-class precision/F1, AUPRC, ECE — at threshold 0.5 and at the val-chosen Youden threshold.
- Model-selection rule: highest mean val AUROC across 3 seeds.
- Baselines: E1, E2, E3. Analysis: mean ± SD over seeds; overlapping seed ranges are reported as "no clear difference" (no significance test claimed with n = 3).

## Experiment matrix
| ID | Experiment | Purpose | Command | Status |
|---|---|---|---|---|
| E0 | Data validation and leakage audit | Establish data integrity | `python -m scripts.validate_data --out reports/data_validation_28.json` | Done (2026-10-18, Actions run 39142842061) |
| E1 | Re-run of MedMNIST v2 ResNet-18 (28) protocol | Verify starting point | `python -m scripts.run_experiments --configs configs/resnet18_medmnist_protocol.yaml --report reports/val_e1.json` | Not run yet (≈4 h CPU; `experiments` workflow with `include_e1`) |
| E2 | Small CNN baseline | Reference performance | `python -m scripts.run_experiments --configs configs/small_cnn.yaml configs/resnet18.yaml configs/small_cnn_classweight.yaml configs/small_cnn_noaug.yaml --report reports/val_summary.json` | Done |
| E3 | ResNet-18 (same recipe as E2) | Controlled model comparison | (same command as E2) | Done |
| E4 | Ablations: (a) class weighting, (b) no augmentation | Test contribution of each training choice | (same command as E2) | Done |
| E5 | Explainability evaluation (val split) | Faithfulness / stability / shortcut checks | `python -m scripts.explain --ckpts runs/<sel>_s0/best.pt runs/<sel>_s1/best.pt runs/<sel>_s2/best.pt --split val --out reports/xai_<sel>_val` | Done (ResNet-18, val) |
| E6 | Robustness / cross-domain | Generalisation | Not feasible: no external dataset available; noise-perturbation stability is covered in E5 | Not planned |
| E7 | Final held-out evaluation | Unbiased assessment | `python -m scripts.final_eval --val-report reports/val_summary.json --final` | Done once (2026-10-18) |

Pipeline verification on synthetic data (not an experiment): all scripts run end-to-end in `tests/test_pipeline.py`.

## Metrics
AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, per-class precision/recall/F1, confusion matrix, ECE (10 bins), percentile bootstrap 95% CI (1,000 resamples of test cases).

## Fair comparison
- Identical splits and preprocessing for all configs (normaliser fitted on train).
- Tuning budget: none — hyper-parameters fixed a priori in `configs/`; no search on val beyond epoch/threshold selection.
- Compute: CPU only (see MODEL_DESIGN.md).
- 3 seeds per config; report mean ± SD and all values.
- Published MedMNIST numbers are compared only with E1/E3 on the same official test split, and differences in protocol are stated.

## Results table
| Run ID | Commit | Dataset/split | Model/config | Seed | Metrics artifact | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | No real-data runs yet: zenodo.org blocked in this environment |

## Reporting
Include negative results, errors, confidence/uncertainty where feasible, and limitations. Never invent missing values.
