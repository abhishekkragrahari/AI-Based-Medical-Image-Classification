# Project Progress Log

Update this file after each meaningful work session. Record actual outcomes only.

## Current status
- Phase: Implementation complete and tested on synthetic data; real-data experiments (E0–E7) not started
- Last updated: 2026-10-18
- Repository URL/commit: https://github.com/abhishekkragrahari/AI-Based-Medical-Image-Classification, branch `main`
- Dataset/version: PneumoniaMNIST, MedMNIST v3.0.2 files (Zenodo record 10519652), 28 px — **not yet downloaded**
- Current blocker: `zenodo.org` returns HTTP 403 from the cloud session's egress proxy (network policy). Fix: add `zenodo.org` to the environment's allowed domains (cloud environment menu → Edit → Network access), or run the commands locally.

## Completed
- [x] Repository inspected (MedMNIST @ 805b742, see REPOSITORY_ANALYSIS.md)
- [x] License and reuse conditions reviewed (code Apache-2.0, data CC BY 4.0)
- [x] Research question defined (RESEARCH_GAP.md; no novelty claimed)
- [ ] Dataset validated — code ready (`scripts/validate_data.py`), blocked on download
- [ ] Splits and leakage checks completed — official splits; check code ready; patient-level check impossible (no IDs)
- [ ] Baseline reproduced — E1 config ready, blocked
- [ ] Candidate models evaluated — E2–E4 configs ready, blocked
- [ ] Explainability analysis completed — code ready, blocked
- [x] Tests passed (39/39, synthetic data)
- [x] One-command pipeline (`scripts/run_all.py`), automatic results report (`scripts/make_report.py`), GitHub Actions CI
- [x] Deployment/inference workflow tested (CLI + Gradio, synthetic weights and images)
- [x] README updated (no results claimed)

## Session log
### 2026-10-18 (cloud session, "complete the project")
- Goal: close the remaining gaps that do not need the real data.
- Added `scripts/run_all.py` (download → E0 → E2–E4 [+E1] → E5 → E7 only with `--final` → report; `--quick` writes to `runs/quick`, `reports/quick`), `scripts/make_report.py` (Markdown from `reports/*.json`, `--update-readme` replaces the README block between `<!-- results:start/end -->`), `.github/workflows/tests.yml`, VS Code launch entries, 2 tests.
- Commands: `python -m pytest -q` → 39 passed; `python -m scripts.run_all --quick` → download fails cleanly with exit 1 (zenodo.org still blocked here).
- Still open: real-data run on the owner's PC (`python -m scripts.run_all --quick`, then `--final`); owner's licence choice.

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
  6. Owner: choose a licence for this repository's code.

## Experiment links
- Configs: `configs/`
- Runs (git-ignored): `runs/<config>_s<seed>/` → `best.pt`, `summary.json`, `history.json`, `val_scores.npy`, `{split}_metrics.json`, `test_access.log`
- Reports (committed, small): `reports/data_validation_28.json`, `reports/val_summary.json`, `reports/val_e1.json`, `reports/test_summary.json`, `reports/xai_<model>_val/`
- None of these exist yet for real data.
