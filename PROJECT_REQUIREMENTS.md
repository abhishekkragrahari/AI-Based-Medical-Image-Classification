# Project Requirements

## Objective
Build a reproducible medical image classification pipeline with explainability, based on a reviewed public repository and an appropriately licensed dataset.

## Functional requirements
- Inspect and document the original repository before editing.
- Load and validate dataset metadata and labels.
- Check for corrupted, duplicate, mislabeled, and potentially leaked samples.
- Provide reproducible train/validation/test splits.
- Train at least one baseline model.
- Evaluate with appropriate metrics and class-wise results.
- Generate explainability outputs where technically appropriate.
- Save model configuration, checkpoints, metrics, and experiment metadata.
- Provide an inference interface and clear setup instructions.

## Non-functional requirements
- Reproducible random seeds and documented environment.
- CPU-compatible minimum workflow where feasible.
- Resource-aware defaults for GPU memory.
- Clear logging and useful error messages.
- No credentials or private data committed to Git.
- Respect dataset and repository licenses.

## Acceptance criteria
- Fresh setup instructions are tested.
- Dataset validation and model tests pass.
- A baseline run can be reproduced from documented configuration.
- Final evaluation uses a held-out test set that was not used for tuning.
- Metrics are computed by scripts, not manually invented.
- Limitations and intended research use are documented.

## Decisions
- Target modality and task: **decided** — pediatric chest X-ray, binary normal vs pneumonia.
- Dataset and applicable terms: **decided** — PneumoniaMNIST (MedMNIST v2), CC BY 4.0.
- Public GitHub repository to build on: **decided** — https://github.com/MedMNIST/MedMNIST (Apache-2.0).
- Compute environment and available hardware: **checked** — Claude Code cloud session, 4 vCPU, 15 GB RAM, no GPU; CPU-only path is the default.

## Decisions still required
- License for this repository's own code (owner's decision).
- Network access to `zenodo.org` in the cloud environment (or another approved way to provide the data) — currently blocking all real-data experiments.
