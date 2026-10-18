# Claude Code Project Instructions

## Mission
Develop an end-to-end AI-based medical image classification project with explainable AI (XAI), using a public GitHub repository as a starting point where appropriate.

## Mandatory workflow
1. Inspect the repository, its license, README, source code, dependencies, data pipeline, and existing tests before making changes.
2. Summarize the current state and propose a staged plan before substantial implementation.
3. Preserve existing functionality unless a documented change is needed.
4. Never assume datasets, files, GPU availability, credentials, or experiment results exist. Check first.
5. Implement in small, reviewable steps and run relevant tests after changes.
6. Record completed work, commands, results, errors, and next steps in `PROGRESS.md`.
7. Keep `README.md` aligned with the actual implementation and verified results.
8. Do not fabricate performance metrics, citations, research novelty, or successful test runs.
9. Avoid data leakage: split by patient or source group when appropriate; fit preprocessing only on training data; keep test data untouched until final evaluation.
10. Keep secrets, private data, and large datasets out of Git. Do not commit model weights or datasets unless the project explicitly permits it and the repository policy supports it.

## Research integrity
- Treat novelty as a hypothesis to investigate, not a claim to assume.
- Compare against meaningful baselines and document the evaluation protocol.
- Report class-wise metrics, uncertainty/limitations, and failure cases where feasible.
- Medical predictions are research outputs, not clinical diagnoses.
- Respect dataset terms, repository licenses, and attribution requirements.

## Hardware awareness
Check available CPU, RAM, GPU, VRAM, and disk space before selecting batch size or model size. Provide a CPU-compatible path and document resource constraints.

## Documentation
Use the following project documents as the source of truth:
- `PROJECT_REQUIREMENTS.md`
- `PROJECT_PLAN.md`
- `REPOSITORY_ANALYSIS.md`
- `RESEARCH_GAP.md`
- `DATASET_GUIDE.md`
- `MODEL_DESIGN.md`
- `EXPERIMENT_PLAN.md`
- `XAI_PLAN.md`
- `TESTING.md`
- `DEPLOYMENT.md`
- `PROGRESS.md`

If instructions conflict, explain the conflict and ask for clarification rather than silently ignoring it.
