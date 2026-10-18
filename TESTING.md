# Testing and Quality Assurance

## Test levels
### Unit tests
- Label parsing and mapping.
- Image loading and transformation.
- Dataset split and grouping logic.
- Metric calculation.
- Checkpoint save/load.
- Prediction shape and output validation.

### Integration tests
- Tiny synthetic or approved sample dataset passes through the full pipeline.
- One short training run completes.
- Evaluation produces expected output files.
- Inference works on valid inputs and handles invalid inputs clearly.
- XAI output generation works for a supported model.

### Data validation
- No missing or invalid labels.
- Expected dimensions/channels.
- No patient/source overlap across splits where identifiers are available.
- No exact duplicates across splits.
- Class distribution is reported.

## Reproducibility
Record Python and library versions, operating system, hardware, random seeds, config, dataset version, and Git commit.

## Safety and security
- Never load untrusted serialized model files without assessing risk.
- Keep secrets out of code and logs.
- Avoid executing downloaded scripts without review.
- Do not include private medical data in test fixtures.

## Release checklist
- [ ] Tests pass in a clean environment.
- [ ] Setup instructions are accurate.
- [ ] Evaluation script uses the intended split.
- [ ] No results are fabricated.
- [ ] Known limitations are documented.
