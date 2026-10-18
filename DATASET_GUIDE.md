# Dataset Guide

## Dataset selection
Choose one dataset aligned with the research question and confirm its license, access conditions, label definitions, and intended use. Examples to investigate include MedMNIST subsets, NIH ChestX-ray14, and other established public datasets; verify current terms from their official sources before use.

## Dataset card
- Name/version: [fill in]
- Official source: [fill in]
- License/terms: [verify]
- Modality and task: [fill in]
- Number of images/patients: [measure from files]
- Label definitions: [fill in]
- Known biases and limitations: [fill in]

## Data integrity checks
- Verify file readability, dimensions, channels, and label values.
- Check class counts and missing metadata.
- Check exact and near duplicates.
- Check patient/source overlap between splits.
- Inspect label quality and uncertain labels.
- Preserve original data and document transformations.

## Splitting
Prefer patient-level or source-level splitting when multiple images may belong to the same patient or source. Keep the test set isolated from model selection. Use stratification where suitable, without violating grouping constraints.

## Preprocessing
- Resize and normalize based on model requirements.
- Fit data-dependent transformations using training data only.
- Apply augmentation to training data only.
- Avoid transformations that alter clinically relevant features.
- Document all settings.

## Data governance
Do not commit datasets, patient information, or access credentials to Git. Follow the dataset's terms and institutional rules.
