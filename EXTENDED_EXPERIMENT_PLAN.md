# Extended Experiment Plan: XAI Faithfulness vs Input Resolution (E10)

Status: **draft, 2026-10-24. Not yet frozen; no E10 run has happened.** Freeze (date + commit) before the first real-data training run. This plan extends `EXPERIMENT_PLAN.md` (E0–E9) and follows from `LITERATURE_NOVELTY_CHECK.md`. Kept as a separate file because open PR #6 also edits `EXPERIMENT_PLAN.md`.

## 1. Research question and hypotheses
**RQ.** Across three MedMNIST+ datasets, does the perturbation faithfulness of saliency explanations (relative to random attribution) change with input resolution (28, 64, 128, 224 px)? Is any change explained by the faithfulness metric itself rather than by the model?

Pre-registered hypotheses (primary endpoint: the **faithfulness gap** = random deletion AUC − method deletion AUC on the validation split; > 0 means better than random):
- **H1 (trend).** The Grad-CAM gap increases with resolution: the per-dataset slope of gap vs log2(resolution) is > 0, with a 95% bootstrap CI above 0, in at least 2 of 3 datasets.
- **H2 (not a metric artefact).** The H1 direction holds under all three fill strategies (§5) and for insertion as well as deletion.
- **H3 (agreement).** Cross-seed top-10% IoU of explanations increases with resolution.
- **Secondary (descriptive, no hypothesis).** Accuracy, calibration (ECE) and compute cost vs resolution. These test-set results are reported once at the end.

Negative results get reported in full. If H1 fails, the paper becomes "faithfulness does not improve with resolution", which is still reportable.

## 2. Datasets (MedMNIST+ official files, Zenodo record 10519652)
| Dataset | Modality | Task | Channels | Train / Val / Test | Licence | Why |
|---|---|---|---|---|---|---|
| PneumoniaMNIST | Paediatric chest X-ray | binary | 1 | 4,708 / 524 / 624 | CC BY 4.0 | Existing results (E0–E9); continuity |
| BreastMNIST | Breast ultrasound | binary | 1 | 546 / 78 / 156 | CC BY 4.0 | Different modality; small (stress-tests CIs) |
| BloodMNIST | Blood cell microscopy | 8-class | 3 (RGB) | 11,959 / 1,712 / 3,421 | CC BY 4.0 | Multi-class, colour, larger |

Counts are from `medmnist.INFO` (v3.0.2) and must be re-measured in E0 for every size. Excluded: DermaMNIST (CC BY-NC 4.0, and HAM10000 has multiple images per lesion, so split leakage); OCTMNIST (97k train images, too costly at 224 px on this laptop).

Known data limitations (must appear in the paper): official splits, no patient IDs, so patient-level leakage can't be ruled out (E0 found 8 PneumoniaMNIST train/val duplicates). The PneumoniaMNIST **test set has already been used** in E7, E8, E1b and E9, so its E10 test numbers are flagged as non-fresh. BreastMNIST and BloodMNIST test sets are untouched and are evaluated once.

## 3. Models and training
- `small_cnn` (287 k params) at all 4 resolutions; `resnet18` (from scratch, small-input stem) at all 4 if the M0 budget allows, otherwise 28–128 px only. The decision is recorded in M0, before any training.
- Same recipe as `configs/small_cnn.yaml` (Adam 1e-3, wd 1e-4, ≤40 epochs, patience 10, augmentation on, no class weighting). Batch size scales with resolution (64 → 64/16/5 at 64/128/224 px) and the learning rate stays fixed. Mixed precision (AMP) is allowed and recorded.
- No ImageNet pretraining: it would make "resolution" interact with the pretraining resolution (224 px).
- **3 seeds** per (dataset, resolution, model). Full grid: 3 × 4 × 2 × 3 = **72 training runs**; small-CNN-only fallback: 36.
- Checkpoint selection on validation only (existing rule).

## 4. Explanation methods and controls
| Item | Setting |
|---|---|
| Grad-CAM | last conv block, existing implementation |
| Integrated Gradients | 32 steps, zero (= train-mean) baseline |
| Occlusion | **patch and stride scaled to image size**: patch = side/7, stride = side/14 (28 px → 4/2 as now; 224 px → 32/16) |
| Random attribution | 5 random rankings per image (existing `random_deletion_auc`) |
| Trivial baselines | centre-Gaussian prior and Sobel edge magnitude. A method that doesn't beat these isn't informative |
| Sanity check | Adebayo et al. cascading model randomisation (top layer → all layers); Spearman between original and randomised maps. Methods that don't degrade are flagged |
| Target class | predicted class; also true class for misclassified images (reported separately) |

## 5. Faithfulness metrics and metric controls (H2)
- **Deletion AUC and insertion AUC**, 20 steps, removing a *fraction* of pixels per step (so the steps are resolution-independent).
- **Fill strategies** (the main known artefact):
  1. `zero`: training mean in normalised space (current E5 setting);
  2. `blur`: Gaussian-blurred copy, σ = side/14;
  3. `noise-impute`: ROAD-style noisy linear imputation (Rong et al., ICML 2022). **Citation from memory, verify before use.**
- **Unit of deletion:** pixel-wise (current) and patch-wise (side/14 patches). At 28 px, single-pixel deletion creates high-frequency patterns the model never saw, while at 224 px it doesn't. Patch-wise deletion removes that asymmetry.
- Other measures: border mass (existing), noise stability (σ = 0.05, existing), cross-seed top-10% IoU (H3).

All XAI analysis runs on the **validation split, 200 images per (dataset, resolution)**, stratified by class (all 78 for BreastMNIST). The same images are used across resolutions (same index, different file size).

## 6. Statistics
- Per (dataset, resolution, method): gap averaged over seeds; image-level percentile bootstrap, 1,000 resamples, 95% CI.
- **Trend (H1/H3):** bootstrap the slope of the per-resolution mean vs log2(size), resampling images within each resolution.
- **Multiplicity:** Holm correction over 3 datasets × 3 methods for H1. H2 is a robustness check with no correction (all-or-nothing criterion).
- Accuracy endpoints: macro AUROC (one-vs-rest for BloodMNIST), balanced accuracy, macro-F1, ECE; seed ensembles with paired bootstrap between resolutions (existing `final_eval` approach).

## 7. Code changes required (not yet done)
The current pipeline is binary-only, single-channel, and has PneumoniaMNIST hard-coded.
1. `medxai/data.py`: download/load any MedMNIST flag and size; keep MD5 checks; RGB support; per-channel normaliser fitted on train.
2. `medxai/train.py`, `medxai/metrics.py`: multi-class (full probability matrix instead of `[:, 1]`), macro AUROC, balanced accuracy. Binary behaviour must stay identical (existing tests guard this).
3. `medxai/xai.py`: resolution-scaled occlusion patch; insertion AUC; `blur` and `noise-impute` fills; patch-wise deletion; trivial baselines; model-randomisation test.
4. Batch size by resolution, AMP flag, `cudnn.benchmark` off by default (it stalled at 128 px on this laptop).
5. `scripts/run_e10.py` grid runner with resume (skip finished runs) for overnight runs; aggregator + report.
6. Tests for each of the above on synthetic data (binary + multi-class, 1 + 3 channels).

## 8. Hardware and compute budget
Machine: Intel i5-10500H (12 threads), 15.8 GB RAM, **RTX 3050 Laptop 4 GB**, 145 GB free disk, Windows 11, PyTorch 2.6.0+cu124 (`.venv`).

Benchmark (`python -m scripts.benchmark_gpu`, synthetic data, 3 channels, training step incl. backward, ms per image):

| Model | 28 px | 64 px | 128 px | 224 px | Peak VRAM |
|---|---|---|---|---|---|
| small_cnn, battery (2026-10-24) | 2.5 | 10.5 | 38.8 | 118.9 | ≤ 0.29 GiB |
| small_cnn, AC power (2026-10-24) | 2.6 | 11.1 | 40.1 | 132.7 | ≤ 0.29 GiB |
| resnet18, battery | 19.6 | 68.9 | 325 | 1,208 | ≤ 1.31 GiB |
| resnet18, AC power | 21.5 | 99.8 | 383 | 1,195 | ≤ 1.31 GiB |

**Both runs were throttled.** `nvidia-smi -q` showed the GPU power limit at **10 W (default 35 W, max 40 W)** and the SM clock pinned at 210 MHz (P8) during the run. Raw throughput measured 0.14 TFLOPS fp32 matmul / 0.15 TFLOPS fp32 conv / 0.45 TFLOPS fp16. Windows' active power-mode overlay was `961cc777-2547-4f9d-8174-7d86181b8a7a` (believed to be "Best power efficiency"; verify in Settings). For comparison, ResNet-18 at 28 px here works out to ≈1.1 h per PneumoniaMNIST seed (40 epochs), slower than the 36–59 min measured on the GitHub CPU runner. **Fix before planning further:** Windows Settings → System → Power & battery → Power mode = *Best performance*; in MSI Center, set the User Scenario to a performance mode. Then rerun the benchmark and add an "unthrottled" row.

**Power-cap log (2026-10-24, still unresolved):**
| Check | Battery | AC + Ultimate Performance plan | After user set "Best performance" (reported) |
|---|---|---|---|
| `Win32_Battery.BatteryStatus` | 1 (discharging) | 2 (AC) | 2 (AC) |
| Active plan (`powercfg /getactivescheme`) | Ultimate Performance | Ultimate Performance | **still Ultimate Performance** |
| Power-mode overlay | `961cc777-…` | `961cc777-…` | `961cc777-…` (unchanged) |
| GPU power limit (`nvidia-smi -q -d POWER`) | 10 W | 10 W | **10 W** (default 35 W) |
| fp32 matmul throughput | — | 0.14 TFLOPS | 0.14 TFLOPS |

Next things to try: set Balanced via Control Panel or `powercfg /setactive 381b4222-f2df-4482-a3b8-f6c3a2ec2b52`, then Power mode = Best performance, then reboot; NVIDIA Control Panel → Power management mode = Prefer maximum performance. Done when `nvidia-smi -q -d POWER` shows a limit above 10 W and SM clocks rise above 210 MHz under load. Until then, all timings in this section are throttled.

**Upper-bound budget with the throttled AC numbers** (all 3 datasets, 17,213 train images, 3 seeds, 40 epochs with no early stopping):

| Model | 28 px | 64 px | 128 px | 224 px | All 4 |
|---|---|---|---|---|---|
| small_cnn | 1.5 h | 6.4 h | 23 h | 76 h | **107 h** |
| resnet18 | 12 h | 57 h | 220 h | 685 h | **975 h** |

Formula per run: `train images × epochs × ms/img`. Early stopping (patience 10) will cut this, but by an unknown amount. XAI evaluation is extra (dominated by occlusion and IG at 224 px).

**Decision on ResNet-18 range (provisional, pending the unthrottled benchmark):**
- Throttled: ResNet-18 can't cover all 4 resolutions (≈975 h, i.e. 40+ days). Even 28–128 px is ≈289 h. Only the small CNN grid (≈107 h, under 5 days of overnight runs) is feasible.
- Reason beyond the GPU: `ResNet18Small` replaces the stride-2 7×7 stem and drops the max-pool, so at 224 px it runs layer1 on 224×224 maps, ≈16× the FLOPs of a standard ResNet-18 at 224 px. Its cost grows with pixel count all the way up. The cost ratio between 224 and 28 px (≈56×) is a property of the architecture and will stay the same once the GPU is unthrottled.
- **Decided:** small CNN at all 4 resolutions (primary H1–H3 analysis); ResNet-18 at 28/64/128 px (secondary architecture check).
- **Open decision (owner, 2026-10-24):** ResNet-18 at 224 px. Add it only if an unthrottled benchmark brings that cell (3 datasets × 3 seeds) under ≈48 h, which needs ≈14× the throttled speed. Must be decided before the plan is frozen (M0). Don't switch to the standard stem at high resolution only: that would change architecture and resolution together and confound H1. The XAI cost at 224 px is dominated by occlusion (13 × 13 = 169 forward passes per image with the scaled patch) and IG (32 passes).

Also observed: with a fixed batch of 64, the 128 px run stalled (VRAM spilling to system RAM on Windows instead of raising OOM). Hence the resolution-scaled batch.

Fallbacks, in order, if the AC budget is too large: (1) ResNet-18 only up to 128 px; (2) BloodMNIST 224 px subsampled to 4,000 train images, with the subsampling documented; (3) GitHub Actions CPU runners for the 28/64 px runs (workflows exist).

## 9. Milestones
| ID | Step | Done when |
|---|---|---|
| M0 | Remove the 10 W GPU power cap (Windows power mode / MSI Center), rerun `scripts.benchmark_gpu`, confirm the GPU leaves P8 under load; finalise the ResNet-18 range | Unthrottled row + budget recorded here; plan frozen with date + commit |
| M1 | Code changes §7 with tests | `pytest` green; binary results unchanged on a PneumoniaMNIST 28 px re-run of one seed |
| M2 | Download 3 datasets × 4 sizes; E0 validation for each | `reports/e10/data_validation_*.json` |
| M3 | Smoke run: 1 seed, 2 epochs, every cell of the grid | No crash, memory within 4 GB, timings logged |
| M4 | Full training grid | 72 (or 36) `summary.json` files |
| M5 | XAI evaluation on val (H1–H3) | `reports/e10/xai_*.json` + figures |
| M6 | One test evaluation (accuracy endpoints) | `reports/e10/test_summary.json`; test access logged |
| M7 | Read TRAX full text; position paper vs TRAX and Kulkarni et al. 2026; draft | Manuscript draft |

## 10. Risks
- TRAX may already vary resolution, which would shrink the contribution (M7 deliberately sits before writing; move it earlier if the PDF becomes available).
- BreastMNIST (78 val / 156 test images) gives wide CIs; results could be inconclusive.
- Resolution changes the effective receptive field of a fixed architecture, so "resolution" and "model capacity relative to image" are confounded. Report this; the two architectures give partial control.
- Grad-CAM can be constant (undefined Spearman) for weak models. Count and report `n_undefined` (already supported).
- Laptop thermals during multi-day runs: use resumable runs and log GPU clock with every run.
