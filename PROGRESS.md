# Project Progress Log

Update this file after each meaningful work session. Record actual outcomes only.

## Current status
- Phase: Real-data experiments E0, E2–E5, E7 complete (2026-10-18); E1 optional and not run
- Last updated: 2026-10-18
- Repository URL/commit: https://github.com/abhishekkragrahari/AI-Based-Medical-Image-Classification, branch `main`
- Dataset/version: PneumoniaMNIST, MedMNIST v3.0.2 files (Zenodo record 10519652), 28 px — downloaded and MD5-verified on the Actions runner
- Current blocker: none. (`zenodo.org` is still blocked in the cloud session; real-data runs use the `experiments` GitHub Actions workflow.)

## Completed
- [x] Repository inspected (MedMNIST @ 805b742, see REPOSITORY_ANALYSIS.md)
- [x] License and reuse conditions reviewed (code Apache-2.0, data CC BY 4.0)
- [x] Research question defined (RESEARCH_GAP.md; no novelty claimed)
- [x] Dataset validated (E0, `reports/data_validation_28.json`)
- [ ] Splits and leakage checks completed — official splits; check code ready; patient-level check impossible (no IDs)
- [ ] Baseline reproduced — E1 config ready, not run yet (E2 small CNN baseline done)
- [x] Candidate models evaluated (E2–E4 val, E7 test once)
- [x] Explainability analysis completed (E5, ResNet-18, val)
- [x] Tests passed (39/39, synthetic data)
- [x] One-command pipeline (`scripts/run_all.py`), automatic results report (`scripts/make_report.py`), GitHub Actions CI
- [x] Deployment/inference workflow tested (CLI + Gradio, synthetic weights and images)
- [x] README updated with script-generated results and their interpretation

## Session log
### 2026-10-20 — E9: CAM-faithfulness regularisation (new method, hypothesis)
- Added `medxai/faithful.py` (CAM deletion + equivariance losses, BatchNorm statistics kept clean during the extra passes), `SmallCNN.features()/cam()`, `augment_batch(return_theta=True)`, new `TrainConfig` fields (default off, so earlier configs train exactly as before), configs `small_cnn_cfr*.yaml`, `scripts/compare_xai.py`, `tests/test_faithful.py` and the `e9-cam-faithfulness` workflow. 54 tests pass locally.
- Measured cost on this 4-core CPU: ≈20 s per epoch with CFR vs ≈7 s without (28 px, 4,708 images).
- Pre-registration written in EXPERIMENT_PLAN.md before the first run.
- Run 39405671236 (≈65 min: training 5–24 min per job, val comparison 25 min, test comparison ≈15 min). Results in `reports/e9/` and README. Primary hypothesis not supported (val Grad-CAM gap −0.005, CI −0.021 to +0.011); AUROC unchanged; deletion-only ablation improves Grad-CAM gap +0.080 but worsens IG, so the gain is specific to the trained explanation. For cfr_eq the mean Grad-CAM border mass is NaN in the JSON because at least one map was constant (border mass undefined); `compare_xai` now ignores undefined values and counts them.

### 2026-10-20 — E1: MedMNIST v2 benchmark comparison
- Added `scripts/aggregate_runs.py` and the `e1-medmnist-protocol` workflow (one job per seed, then a one-time test evaluation). Run 39362050921: ≈2.6 h per seed; test AUROC 0.949 ± 0.011, accuracy 0.868 ± 0.026, vs published ResNet-18 (28) 0.944 / 0.854 (arXiv 2110.14795).
- Paired comparison (compare run 39376661730): this repo's ResNet-18 recipe minus E1, AUROC +0.022 (95% CI +0.012 to +0.034); copied from the job log to `reports/compare_resnet18_recipe_vs_e1.json`.

### 2026-10-19 — Stronger results: 64 px, paired tests, de-duplicated val
- Added `run_all --size` (and the `size` input of the `experiments` workflow), validation metrics/threshold without train duplicates, seed ensembles and a paired bootstrap of AUROC differences in `final_eval`, and `scripts/compare_sizes.py` (+ `compare-sizes` workflow). `python -m pytest -q` → 45 passed.
- Run 39296675487 (64 px, small CNN ×2 configs × 3 seeds, ~1 h 25 min): test specificity 0.870 ± 0.071 vs 0.751 ± 0.024 at 28 px; paired ensemble difference +0.099 (95% CI +0.058 to +0.141); AUROC difference +0.003 (CI −0.004 to +0.011). Occlusion and Grad-CAM now beat random deletion. Second look at the test set (decided after the 28 px results).
- Run 39296680857 (28 px re-run, ~3 h 25 min): per-run test metrics identical to run 39142842061 (reproducible on the CPU runner).
- `reports/compare_28_vs_64.json` was copied from the job log of compare run 39325733133 (artifact downloads are blocked from the cloud session).
- Next: E1 (MedMNIST protocol) if wanted; external validation would need another dataset.

### 2026-10-19 — In-browser demo
- Added `scripts/export_web_model.py` (BatchNorm folded into the convs, float32 base64 JSON, a few **validation** images as samples), `web/engine.js` (plain-JS forward pass, Pillow-style bicubic resize, occlusion with patch 4 / stride 2), `web/demo_template.html` and `scripts/build_web_demo.py`.
- Tests: `tests/test_web_export.py` checks the NumPy reference and the JS engine (run under node) against PyTorch: identical preprocessing, probabilities within 1e-4, heatmap within 1e-3. `python -m pytest -q` → 42 passed.
- Workflow `web-demo-model` trains `configs/small_cnn.yaml` seed 0 on the real data, exports and builds the page, and pushes it to a throwaway orphan branch `web-model/<run id>` (artifact downloads are blocked from the cloud session). Weights are not committed to `main`.
- Occlusion takes about 10 s per image in headless Chromium on the cloud machine.

### 2026-10-18 (real-data run via GitHub Actions)
- zenodo.org blocked in the cloud session, so added `.github/workflows/experiments.yml` (PR #4) and ran it: run 39142842061, `python -m scripts.run_all --final`, under 3.5 h on a GitHub CPU runner (ResNet-18 ≈36–59 min per seed).
- Results committed from branch `results/39142842061`: `reports/` + README Results (generated by `make_report`). Interpretation added to README ("What the results show").
- Key outcomes: val AUROC ≈0.998 for all models, test AUROC 0.965–0.980; test specificity ≈0.75 at 0.5 → val optimistic (8 train/val duplicates). ResNet-18 selected by rule but no clear difference from small CNN. Occlusion beats random deletion; Grad-CAM/IG do not.
- Next: optional E1 run; external validation not possible with available data.

### 2026-10-18 (cloud session, "complete the project")
- Goal: close the remaining gaps that do not need the real data.
- Added `scripts/run_all.py` (download → E0 → E2–E4 [+E1] → E5 → E7 only with `--final` → report; `--quick` writes to `runs/quick`, `reports/quick`), `scripts/make_report.py` (Markdown from `reports/*.json`, `--update-readme` replaces the README block between `<!-- results:start/end -->`), `.github/workflows/tests.yml`, VS Code launch entries, 2 tests.
- Commands: `python -m pytest -q` → 39 passed; `python -m scripts.run_all --quick` → download fails cleanly with exit 1 (zenodo.org still blocked here).
- Owner chose the MIT licence for the code → `LICENSE` added.
- Still open: real-data run on the owner's PC (`python -m scripts.run_all --quick`, then `--final`).

### 2026-10-18 (cloud session)
- Goal: carry out Stages 1–8 of PROJECT_PLAN.md as far as possible in the cloud session.
- Hardware checked: 4 vCPU Intel Xeon 2.1 GHz, 15 GB RAM, no GPU (`nvidia-smi` absent), ~31 GB free disk, Python 3.13.16, Linux 6.18 x86_64.
- Files changed: `medxai/` (data, models, train, metrics, evaluate, xai, infer), `scripts/` (download_data, validate_data, train, run_experiments, evaluate, final_eval, explain, predict), `configs/*.yaml`, `tests/*`, `app.py`, `requirements.txt`, `pytest.ini`, `.gitignore`, all planning docs.
- Commands run:
  - `git clone https://github.com/MedMNIST/MedMNIST.git` (reviewed at 805b742)
  - `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` → failed (403, host blocked); `pip install torch torchvision` from PyPI succeeded (torch 2.14.1+cu130 wheel, runs on CPU)
  - `pip install medmnist==3.0.2 scikit-learn matplotlib pytest pyyaml gradio`
  - `curl https://zenodo.org/records/10519652/files/pneumoniamnist{,_64,_128}.npz` → `CONNECT tunnel failed, response 403`
  - `python -m pytest -q` → 37 passed
  - Synthetic timing benchmark (4,708 synthetic 28 px images, 1 epoch): small CNN 4.8 s, ResNet-18 47.8 s
  - README command walkthrough on a synthetic file: validate, run_experiments (2 seeds × 2 epochs), explain, final_eval (refuses without `--final`, exit 2), predict, Gradio app (HTTP 200) — all OK; download_data exits 1 with a clear message
- Tested versions: torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, pillow 12.3.0, PyYAML 6.0.3, pytest 9.1.1, gradio 6.30.0.
- Test/experiment results: no real-data results. Synthetic runs only verify that the code works; their metrics mean nothing and are not reported.
- Errors and resolutions:
  - Grad-CAM can be all-zero for a barely trained model → Spearman undefined; now returns NaN and statistics count undefined cases (`n_undefined`).
  - Test expected 2.8 px border; implementation rounds to 3 px → test corrected; uniform reference now computed from image size (0.383 at 28 px).
  - Gradio contacted `huggingface.co` / `api.gradio.app` (telemetry) on launch → disabled via `GRADIO_ANALYTICS_ENABLED=False`.
  - doi.org / arxiv.org / crossref blocked; citations verified via web search instead (RESEARCH_GAP.md marks which).
- Decisions:
  - Official MedMNIST splits kept (comparability); train/val patient overlap documented as a limitation.
  - No ImageNet weights (blocked host, extra licence); ResNet-18 trained from scratch with a small-input stem.
  - No horizontal flip augmentation (chest anatomy is asymmetric).
  - Test set evaluated once for all runs (`final_eval.py`), with the winner chosen on val only.
- Next steps (once `zenodo.org` is reachable or run locally):
  1. `python -m scripts.download_data --size 28`
  2. E0: `python -m scripts.validate_data --out reports/data_validation_28.json` → fill the dataset card counts in DATASET_GUIDE.md
  3. E2–E4: `run_experiments` with the four configs (~45 min CPU); E1 (~4 h CPU)
  4. E5: `scripts/explain.py` on the selected config's 3 seeds (val split)
  5. E7: `scripts/final_eval.py --final` once; then write README Results + limitations/failure cases

## Experiment links
- Configs: `configs/`
- Runs (git-ignored): `runs/<config>_s<seed>/` → `best.pt`, `summary.json`, `history.json`, `val_scores.npy`, `{split}_metrics.json`, `test_access.log`
- Reports (committed, small): `reports/data_validation_28.json`, `reports/val_summary.json`, `reports/val_e1.json`, `reports/test_summary.json`, `reports/xai_<model>_val/`
- None of these exist yet for real data.
