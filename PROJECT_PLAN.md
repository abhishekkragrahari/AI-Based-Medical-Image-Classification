# Master Project Plan

## Stage 0 — Define scope
Choose one medical imaging modality, task, dataset, and target use case. Avoid mixing unrelated modalities without a clear research rationale.

## Stage 1 — Repository reconnaissance
- Record repository URL, commit/version, license, dependencies, entry points, and existing results.
- Map the codebase and run existing tests if safe.
- Identify missing files, data assumptions, known issues, and security concerns.
- Do not overwrite or reimplement the project before understanding it.

## Stage 2 — Research question
- Review recent and foundational literature.
- Define a testable research question and candidate contribution.
- Compare against existing work; do not claim novelty until supported by literature review.
- Predefine primary metrics and evaluation protocol.

## Stage 3 — Data pipeline
- Confirm dataset license, access conditions, labels, patient/source identifiers, and class balance.
- Audit duplicates and possible leakage.
- Define patient-level or group-level splits when relevant.
- Fit transformations and normalization using training data only.
- Document all exclusions and preprocessing choices.

## Stage 4 — Baseline
- Reproduce an existing baseline before proposing improvements.
- Validate the pipeline with a small smoke test.
- Save configurations, logs, checkpoints, and metrics.

## Stage 5 — Models and experiments
- Add selected model families only when justified.
- Use controlled comparisons with the same data splits and comparable tuning budgets.
- Run ablations for proposed components.
- Report multiple seeds or uncertainty estimates where feasible.

## Stage 6 — Explainability
- Select explanation methods compatible with the model and modality.
- Generate examples for correct and incorrect predictions.
- Avoid presenting heatmaps as proof that a model reasons clinically correctly.
- Evaluate explanation stability or localization against appropriate references if available.

## Stage 7 — Testing and hardening
- Run unit, integration, data-validation, and inference tests.
- Test empty inputs, invalid labels, missing files, and CPU fallback where applicable.
- Check dependency and license compatibility.

## Stage 8 — Delivery
- Provide a documented inference workflow or lightweight demo.
- Update README with verified installation, commands, results, and limitations.
- Ensure no private data, secrets, or unapproved artifacts are committed.

## Stage 9 — Research reporting
- Report methods, data splits, baselines, metrics, confidence intervals where possible, limitations, and reproducibility details.
- Do not write results or claims before experiments are complete.

## Completion definition
The project is complete when setup, data checks, baseline, final evaluation, explainability, tests, and documentation are reproducible by another person.
