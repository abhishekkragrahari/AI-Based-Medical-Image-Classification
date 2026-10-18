# Public Repository Analysis

## Repository details
- Repository URL: https://github.com/MedMNIST/MedMNIST
- Owner/project: MedMNIST team (Jiancheng Yang et al.) — `medmnist` Python package
- Branch and commit hash: `main` @ `805b74237fb5bd2db376fd94e50646b6a8905d9d` (commit date 2025-01-11); package version `3.0.2` (`medmnist/info.py`)
- License file and license type: `LICENSE` — Apache License 2.0 (code). Datasets: CC BY 4.0, except DermaMNIST (CC BY-NC 4.0), per the upstream README "License and Citation" section.
- Date reviewed: 2026-10-18

## Review checklist
- [x] Read README and contribution/license terms.
- [x] Inspect directory structure and entry points.
- [x] Review requirements/environment files.
- [x] Identify data download and preprocessing logic.
- [x] Identify train, validation, test, and inference workflows.
- [x] Inspect checkpoint loading and serialization.
- [~] Run existing tests or a documented smoke test. — Upstream has no unit tests; its `python -m medmnist test` command downloads every dataset, which is not possible here (Zenodo is blocked by this environment's network policy). The package was installed (`pip install medmnist==3.0.2`) and imports successfully.
- [ ] Check open issues and known limitations. — Not done: this session's GitHub access is scoped to the project repository only. To do manually.
- [x] Review dependency security and abandoned packages. — Dependencies are mainstream (numpy, pandas, scikit-learn, scikit-image, tqdm, Pillow, fire, torch, torchvision); none pinned. No CVE scan was run.
- [x] Identify hard-coded paths, credentials, and unsafe operations. — See findings.

## Architecture map
| Module | Role | Inputs → outputs |
|---|---|---|
| `medmnist/info.py` | Metadata registry `INFO` (description, URLs, MD5s, task, labels, split sizes, license) per dataset; default root `~/.medmnist` | — |
| `medmnist/dataset.py` | `MedMNIST2D/3D` torch `Dataset`s; `download()` via `torchvision.datasets.utils.download_url` with MD5; loads `{split}_images/labels` from `.npz` | `.npz` → PIL image, int label |
| `medmnist/evaluator.py` | `Evaluator`: AUC and ACC (threshold 0.5) per task type; standard result-file naming | scores → `Metrics(AUC, ACC)` |
| `medmnist/__main__.py` | CLI (`available`, `download`, `clean`, `info`, `save`, `evaluate`, `test`) via `fire` | — |
| `medmnist/utils.py` | Montage/GIF/PNG export helpers | arrays → images |
| `examples/*.ipynb` | Getting-started notebooks (with and without PyTorch) | — |

There is **no training code** in the package; the MedMNIST v2 benchmark training scripts live in a separate experiments repository (not reviewed here). Upstream provides fixed train/val/test splits and an evaluator only.

## Reproducibility status
- Commands run: `git clone https://github.com/MedMNIST/MedMNIST.git`; `pip install medmnist==3.0.2`; source reading of all modules.
- Environment: Python 3.13.16, Linux x86_64, 4 vCPU, 15 GB RAM, no GPU.
- Data download: **blocked** — `zenodo.org` returns HTTP 403 from this session's egress proxy (network policy), so no published result has been reproduced yet.
- Published results: the MedMNIST v2 paper (Sci Data 2023, DOI 10.1038/s41597-022-01721-8) reports ResNet/AutoML baselines on PneumoniaMNIST. These are *repository/paper claims* and have **not** been independently verified here. E1 (`configs/resnet18_medmnist_protocol.yaml`) approximates the described protocol; it is not an exact re-implementation because the training code was not reviewed.

## Reuse and attribution
- Code: Apache-2.0. This project does **not** copy upstream source code; it re-implements a minimal `.npz` loader (`medxai/data.py`) and reuses only factual metadata (Zenodo URLs, MD5 checksums, split sizes, label names). If upstream code is vendored later, keep its `LICENSE` and add a NOTICE entry.
- Data: PneumoniaMNIST is CC BY 4.0 — attribution required. Cite MedMNIST v2 (and v1, ISBI 2021) and the source dataset (Kermany et al., Cell 2018, DOI 10.1016/j.cell.2018.02.010) as requested by upstream.
- The license for *this* repository's new code has not been chosen yet (decision for the owner; see PROJECT_REQUIREMENTS.md).

## Findings
### Strengths
- Standardised, versioned data with published MD5 checksums and fixed official splits — enables comparable evaluation.
- Small, readable code base; permissive licence; CPU-friendly image sizes (28/64/128/224).

### Limitations and risks
1. **No patient identifiers.** The PneumoniaMNIST train/val split was made by splitting the source training set 9:1 (per `info.py` description); multiple images of one patient may fall in both train and val, so validation scores may be optimistic. The test split is the source's separate split, but patient independence cannot be verified from the `.npz`.
2. `download()` and the evaluator catch all exceptions with bare `except:`, hiding root causes.
3. `Evaluator` reports only AUC and ACC@0.5 — no class-wise metrics, calibration, or uncertainty.
4. Default root `~/.medmnist` is created at import time (side effect).
5. `save_results` uses `DataFrame.append`, removed in pandas 2.x (deprecated helper only).
6. `np.load` is called without `allow_pickle=False` explicitly (numpy's default is already `False`, so low risk).
7. Unpinned dependencies.

### Recommended changes (adopted in `medxai/`)
1. High — report class-wise sensitivity/specificity, balanced accuracy, AUPRC, ECE and bootstrap CIs (`medxai/metrics.py`).
2. High — explicit data-integrity/leakage audit incl. exact and near-duplicate checks across splits (`medxai/data.py::validate_dataset`), and document the missing patient-level grouping as a limitation.
3. High — gate test-split access behind an explicit `--final` flag and log every access (`medxai/evaluate.py`).
4. Medium — verify MD5 on download; surface clear errors instead of bare `except`.
5. Medium — load checkpoints with `torch.load(weights_only=True)`.
