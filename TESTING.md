# Testing and Quality Assurance

Run: `python -m pytest -q` (≈ 10 s on 4 CPU threads). All tests use **synthetic** images (`medxai.data.make_synthetic_npz`); no medical data is needed or included.

## Test levels
### Unit tests
- [x] Label parsing and mapping — `tests/test_data.py::test_load_and_shapes`, invalid labels in `test_validation_detects_problems`.
- [x] Image loading and transformation — normaliser fitted on train only (`test_normalizer_fit_on_train_only`), deterministic augmentation (`test_dataset_and_augment`).
- [x] Dataset split and grouping logic — official splits are used as-is; cross-split duplicate detection tested. Grouping by patient: not applicable (no IDs).
- [x] Metric calculation — `tests/test_metrics.py` (hand-computed confusion/class-wise values, AUROC edge cases, ECE, Youden, bootstrap CI).
- [x] Checkpoint save/load — `tests/test_models.py::test_checkpoint_roundtrip`; malicious pickle rejected (`test_rejects_pickled_code`).
- [x] Prediction shape and output validation — `test_output_shape` (28 and 64 px), probabilities sum to 1 in inference test.

### Integration tests (`tests/test_pipeline.py`)
- [x] Tiny synthetic dataset passes through the full pipeline (train → evaluate → explain → predict).
- [x] One short training run completes and is bit-reproducible with the same seed on CPU.
- [x] Evaluation produces expected output files; test split refused without `final=True` / `--final`; test access logged.
- [x] Inference works on valid inputs (grayscale PNG, grayscale-as-RGB JPEG) and rejects invalid ones (colour, too small, wrong extension, corrupt, empty, missing file, missing checkpoint).
- [x] XAI output generation works for Grad-CAM, IG and occlusion; quality metrics are bounded.
- [x] Experiment matrix + final evaluation scripts; all repo configs parse; unknown config keys rejected.
- [x] Gradio app builds with and without weights (`tests/test_app.py`).

### Data validation (`scripts/validate_data.py`)
- [x] No missing or invalid labels.
- [x] Expected dimensions/channels and official split sizes.
- [ ] No patient/source overlap across splits — **cannot be checked**: no identifiers.
- [x] No exact duplicates across splits (+ near-duplicate candidates).
- [x] Class distribution is reported.
Real-data run: **pending** (download blocked).

## Latest result
2026-10-18, Python 3.13.16, torch 2.14.1 (CPU), Linux x86_64: `37 passed`.

## Reproducibility
Each run's `summary.json` records Python/library versions, OS, CPU count, threads, seed, full config, dataset MD5 and Git commit (+ dirty flag).

## Safety and security
- Checkpoints are loaded with `torch.load(weights_only=True)`; `np.load(allow_pickle=False)`.
- No secrets in code or logs; no network calls except `scripts/download_data.py` (official Zenodo URL, MD5-verified).
- No private medical data in test fixtures.

## Release checklist
- [x] Tests pass in a clean environment (fresh venv in this cloud session).
- [x] Setup instructions are accurate (README commands run in this session, except data download).
- [x] Evaluation script uses the intended split (gated test).
- [x] No results are fabricated.
- [x] Known limitations are documented.
