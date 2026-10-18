# Deployment Guide

## Scope
Provide a research demonstration, not a clinical diagnostic service.

## Deployment options
- [x] Command-line inference: `python -m scripts.predict --ckpt runs/<run>/best.pt --image xray.png [--explain gradcam --out heat.png]`
- [x] Local Gradio interface: `python app.py --ckpt runs/<run>/best.pt` (binds to 127.0.0.1 only, `share=False`).
- API: not provided (no integration requirement).

## Required behavior (implemented in `medxai/infer.py`)
- [x] Validate file type (png/jpg/jpeg/bmp/tif/tiff), size (≤ 20 MB, non-empty), dimensions (16–8192 px), colour (rejects true-colour images; accepts grayscale stored as RGB; 16-bit grayscale rescaled).
- [x] Apply the training preprocessing: centre-crop to square, resize to the checkpoint's input size, training-set mean/std stored in the checkpoint. Caveat: the exact resampling kernel MedMNIST used is not verified, so external images may differ slightly from the training distribution.
- [x] Return class scores, labels at 0.5 and at the val-chosen threshold, model name, checkpoint epoch, package version.
- [x] Explicit research-only notice in every output and in the UI.
- [x] Missing weights and invalid images give clear errors (CLI exit code 2; UI error message).
- [x] Uploaded images are processed in memory and not stored.

## Packaging
- Python 3.13.16 (tested); dependencies in `requirements.txt` (lower bounds; tested versions in PROGRESS.md).
- Model checkpoints: **not distributed** — produce them with `scripts/run_experiments.py`. Checkpoint format: `{"state_dict", "meta"}` loaded with `weights_only=True`.
- Setup: see README "Quick start".

## Privacy and security
No external uploads; no logging of image content; no debug mode; no secrets.

## Performance
Measured 2026-10-18 on this cloud session (4 vCPU Intel Xeon 2.1 GHz, no GPU), 1024×1024 PNG input, mean of 10 calls including decoding and preprocessing. Weights came from 1-epoch **synthetic-data** runs; latency does not depend on what the model learned.

| Model | Predict only | + Grad-CAM | + IG (32 steps) | + Occlusion |
|---|---|---|---|---|
| Small CNN | 13.8 ms | 86.1 ms | 40.5 ms | 71.3 ms |
| ResNet-18 | 24.7 ms | 59.1 ms | 237.4 ms | 544.1 ms |

## Release checklist
- [x] Inference tested on representative valid inputs (synthetic images only; real X-ray pending data access).
- [x] Invalid inputs handled.
- [x] Model and preprocessing versions match (normaliser and input size stored in checkpoint; size mismatch rejected).
- [x] Limitations and intended use visible.
- [~] Installation and run commands verified — all except the data download (blocked here).
