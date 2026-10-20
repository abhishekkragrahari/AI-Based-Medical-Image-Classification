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
| E1 | Re-run of MedMNIST v2 ResNet-18 (28) protocol | Verify starting point | `python -m scripts.run_experiments --configs configs/resnet18_medmnist_protocol.yaml --report reports/val_e1.json` | Done 2026-10-20 (`e1-medmnist-protocol` workflow run 39362050921, one job per seed, ≈2.6 h each) → `reports/e1/` |
| E2 | Small CNN baseline | Reference performance | `python -m scripts.run_experiments --configs configs/small_cnn.yaml configs/resnet18.yaml configs/small_cnn_classweight.yaml configs/small_cnn_noaug.yaml --report reports/val_summary.json` | Done |
| E3 | ResNet-18 (same recipe as E2) | Controlled model comparison | (same command as E2) | Done |
| E4 | Ablations: (a) class weighting, (b) no augmentation | Test contribution of each training choice | (same command as E2) | Done |
| E5 | Explainability evaluation (val split) | Faithfulness / stability / shortcut checks | `python -m scripts.explain --ckpts runs/<sel>_s0/best.pt runs/<sel>_s1/best.pt runs/<sel>_s2/best.pt --split val --out reports/xai_<sel>_val` | Done (ResNet-18, val) |
| E6 | Robustness / cross-domain | Generalisation | Not feasible: no external dataset available; noise-perturbation stability is covered in E5 | Not planned |
| E7 | Final held-out evaluation | Unbiased assessment | `python -m scripts.final_eval --val-report reports/val_summary.json --final` | Done once (2026-10-18); re-run 2026-10-19 (run 39296680857) with identical per-run results, adding seed ensembles and paired bootstrap |
| E8 | Input resolution 64 px (small CNN, class-weighted small CNN) | Does resolution fix the low test specificity? | `python -m scripts.run_all --size 64 --final` (`experiments` workflow, size 64) | Done 2026-10-19 (run 39296675487). Added after the 28 px test results were seen, so its test numbers are a **second look** at the test set |
| E1b | Paired test comparison: E3 ResNet-18 (this repo's recipe) vs E1 (MedMNIST protocol) seed ensembles | Does the training recipe matter? | `compare-sizes` workflow, second step | Done 2026-10-20 (run 39376661730) → `reports/compare_resnet18_recipe_vs_e1.json` |
| E8b | Paired 28 px vs 64 px test comparison (small CNN seed ensembles) | Is the 64 px change real? | `python -m scripts.compare_sizes ...` (`compare-sizes` workflow) | Done 2026-10-19 (run 39325733133) → `reports/compare_28_vs_64.json` |
| E9 | CAM-faithfulness regularisation (CFR) for the small CNN, 28 px, with deletion-only and equivariance-only ablations | Can training make explanations faithful at 28 px without losing accuracy? | `e9-cam-faithfulness` workflow (`scripts.run_experiments` + `scripts.compare_xai`) | Done 2026-10-20 (run 39405671236) → `reports/e9/`. Primary hypothesis **not supported**; see README |

Pipeline verification on synthetic data (not an experiment): all scripts run end-to-end in `tests/test_pipeline.py`.

## E9 pre-registration (written 2026-10-20, before any E9 run)
- Motivation: at 28 px the explanations were not better than random deletion (ResNet-18 Grad-CAM deletion AUC 0.751 vs random 0.698, `reports/xai_resnet18_val`); at 64 px Grad-CAM and occlusion were (`reports/size64/xai_small_cnn_val`).
- Method (`medxai/faithful.py`): loss = CE + 1.0 x deletion loss + 0.5 x equivariance loss. Deletion loss: remove the top 20% pixels of the true-class CAM (filled with a 5x5 blurred copy) and penalise `relu(p_y - 0.5)`. Equivariance loss: L1 between the CAM of the augmented image and the augmented CAM of the clean image. Weights fixed a priori, not tuned. Related work: GAIN (Li et al., CVPR 2018), SEAM (Wang et al., CVPR 2020); novelty is a hypothesis pending a literature search.
- Groups (3 seeds each, same recipe as `configs/small_cnn.yaml`): baseline `small_cnn`, `small_cnn_cfr`, `small_cnn_cfr_del`, `small_cnn_cfr_eq`.
- Primary endpoint (validation, all 524 images): Grad-CAM deletion gap = random deletion AUC - Grad-CAM deletion AUC (fill 0, as in E5), averaged over seeds, CFR minus baseline, paired bootstrap 95% CI over images. H1 supported if the CI is above 0.
- Guard against circularity: Grad-CAM is what CFR trains, so occlusion and Integrated Gradients gaps are reported as independent secondary checks; training fill (blur) differs from evaluation fill (0) so the model cannot simply learn to react to the fill; random deletion AUC itself is reported per group.
- Non-inferiority: seed-ensemble val AUROC difference (CFR - baseline) with 95% CI lower bound above -0.01. Secondary: specificity, sensitivity, balanced accuracy at 0.5, border mass.
- Outcome (added after the run): val Grad-CAM gap CFR − baseline −0.005 (95% CI −0.021 to +0.011), H1 not supported; non-inferiority met (AUROC +0.0002, CI −0.0006 to +0.0012). Test: Grad-CAM gap −0.012 (−0.023 to −0.000). Exploratory ablations: deletion-only +0.080 Grad-CAM gap but IG −0.022 (circularity concern); equivariance-only the reverse.
- Test: one comparison only, `cfr` vs `baseline`, same metrics, run in the same workflow after the validation step; no decision is changed after seeing it. This is a further look at a test set that E7/E8/E1b already used, which is a limitation.

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
