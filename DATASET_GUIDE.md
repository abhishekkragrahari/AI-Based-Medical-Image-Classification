# Dataset Guide

## Dataset selection
**PneumoniaMNIST (MedMNIST v2)** — binary pediatric chest X-ray classification (normal vs pneumonia). Chosen because it is openly licensed (CC BY 4.0), small enough for a CPU-only workflow, and has fixed official splits that allow comparison with published baselines. Terms were read from the upstream README and `medmnist/info.py` (v3.0.2) on 2026-10-18.

## Dataset card
- Name/version: PneumoniaMNIST, MedMNIST v2 / package v3.0.2, Zenodo record 10519652. Sizes 28 (default), 64, 128, 224 px.
- Official source: https://github.com/MedMNIST/MedMNIST ; files: `https://zenodo.org/records/10519652/files/pneumoniamnist{,_64,_128,_224}.npz`
- MD5 (from upstream `info.py`): 28 px `28209eda62fecd6e6a2d98b1501bb15f`; 64 px `8f4eceb4ccffa70c672198ea285246c6`; 128 px `05b46931834c231683c68f40c47b2971`; 224 px `d6a3c71de1b945ea11211b03746c1fe1`
- License/terms: CC BY 4.0 (attribution required). Source dataset: Kermany et al., Cell 2018 (DOI 10.1016/j.cell.2018.02.010).
- Modality and task: pediatric chest radiographs (grayscale), binary classification; label 0 = normal, 1 = pneumonia.
- Number of images/patients: documented split sizes train 4,708 / val 524 / test 624 (5,856 total). **Not yet measured from files** — the download is blocked in the current environment (see PROGRESS.md). Patient count: unknown (no identifiers in the release).
- Label definitions: per upstream: "binary-class classification of pneumonia against normal". Bacterial and viral pneumonia are merged into one class in this release.
- Preprocessing by upstream: source images (384–2,916 × 127–2,713 px) were centre-cropped and resized (to 28 px for the default file).
- Known biases and limitations:
  - Single-source pediatric cohort; results may not transfer to adults or other hospitals/scanners.
  - Class imbalance is expected (to be measured by E0).
  - No patient IDs → train/val patient overlap cannot be ruled out (val split was carved from the source training set by image).
  - Very low resolution at 28 px; fine radiological findings are not resolvable.
  - Possible shortcut cues (text markers, borders, positioning) — checked by XAI border-mass analysis (XAI_PLAN.md).

## Data integrity checks
Implemented in `medxai/data.py::validate_dataset`, run by `python -m scripts.validate_data --data data/pneumoniamnist.npz --out reports/data_validation_28.json`:
- [x] Required keys, readability, dtype `uint8`, `N×H×W` grayscale shape, labels ∈ {0,1}.
- [x] Split sizes vs. official counts; empty splits.
- [x] Class counts, pixel statistics, constant (blank) images.
- [x] Exact duplicates within each split (SHA-1 of pixels) and duplicate groups with conflicting labels.
- [x] Exact duplicates across splits.
- [x] Near-duplicate *candidates* across splits (8×8 block-mean, 4-bit quantised key) — for manual review, not proof.
- [ ] Patient/source overlap — **not possible** (no identifiers); documented as a limitation.
- [ ] Label quality review — not possible beyond duplicate-label conflicts; labels are taken as released.

Results: pending data access (E0).

## Splitting
Official MedMNIST splits are used unchanged so that results are comparable with published work. The test split is used once, at the end (E7), through `scripts/final_eval.py --final`; all other code paths refuse it. Patient-level re-splitting is impossible without identifiers.

## Preprocessing
- Images kept at native release resolution (28 px default); no further resizing for training.
- Intensity normalisation: global mean/std **fitted on the training split only** (`Normalizer.fit`), stored in each checkpoint and re-applied at evaluation/inference.
- Augmentation (train only): random rotation ≤ 10° and translation ≤ 2 px. **No horizontal flips** (cardiac position makes chest X-rays left/right asymmetric).
- Inference on external images: centre-crop to square + resize to the checkpoint's input size (mirrors upstream's documented crop+resize; exact upstream resampling kernel not verified).

## Data governance
`data/`, `*.npz`, `runs/`, `*.pt` are git-ignored. Do not commit datasets, patient information, or credentials. Tests use synthetic images only (`make_synthetic_npz`).

## How to obtain the data
```bash
python -m scripts.download_data --size 28          # verifies MD5
# or download manually from the Zenodo URL above into data/
```
In a Claude Code cloud session, `zenodo.org` must be on the environment's allowed-domains list.
