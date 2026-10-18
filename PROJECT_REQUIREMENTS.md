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

## Decisions still required
- Target modality and task (e.g., chest X-ray, histopathology, dermoscopy).
- Dataset and applicable terms.
- Public GitHub repository to build on.
- Compute environment and available hardware.
