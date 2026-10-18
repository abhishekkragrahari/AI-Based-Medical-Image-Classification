# Deployment Guide

## Scope
Provide a research demonstration, not a clinical diagnostic service.

## Deployment options
- Command-line inference script for reproducibility.
- Local Streamlit or Gradio interface for a lightweight demo.
- API only if a real integration requirement exists.

## Required behavior
- Validate file type, size, dimensions, and color channels.
- Apply exactly the preprocessing used during training.
- Return class scores and model/version metadata.
- Display an explicit research-only notice.
- Handle missing model weights and invalid images gracefully.
- Avoid storing uploaded images by default.

## Packaging
Document Python version, dependencies, model checkpoint source, configuration, and setup commands. Pin or constrain dependencies where practical.

## Privacy and security
Do not upload patient images to external services without authorization. Avoid logging sensitive image contents or identifiers. Do not expose debug mode or secrets.

## Performance
Measure inference time and resource usage on the target hardware; do not make unmeasured speed claims.

## Release checklist
- [ ] Inference tested on representative valid inputs.
- [ ] Invalid inputs handled.
- [ ] Model and preprocessing versions match.
- [ ] Limitations and intended use visible.
- [ ] Installation and run commands verified.
