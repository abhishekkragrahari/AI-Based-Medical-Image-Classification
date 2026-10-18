# Model Design

## Design principle
Start with a simple, reproducible baseline. Add complexity only when it tests a defined hypothesis.

## Candidate models
| Model | Code | Params | Status |
|---|---|---|---|
| Small CNN (3 double-conv blocks, 32/64/128 ch, BN, GAP, dropout 0.2) | `medxai/models.py::SmallCNN` | 287,138 | Implemented |
| ResNet-18, from scratch, 3×3 stride-1 stem, no max-pool (small-input variant) | `medxai/models.py::ResNet18Small` | 11,168,706 | Implemented |
| EfficientNet / ViT | — | — | Not planned: no hypothesis needs them, 28 px input, CPU only |

No ImageNet-pretrained weights are used (they are hosted on download.pytorch.org, which is also blocked here, and they would add a weight licence to track). Pretraining is a possible future ablation.

## Pipeline
1. Input validation — `scripts/validate_data.py` (dataset), `medxai/infer.py::validate_and_load` (single images).
2. Dataset loading and transformations — `medxai/data.py`.
3. Model initialization — `medxai/models.py::build_model`.
4. Training and validation — `medxai/train.py`.
5. Checkpoint selection using validation data only — best val AUROC epoch; decision threshold chosen on val (Youden J) as a secondary operating point.
6. One final evaluation on the held-out test set — `scripts/final_eval.py --final` (all runs reported, not just the selected one).
7. Prediction and explanation generation — `scripts/predict.py`, `scripts/explain.py`, `app.py`.

## Training configuration
| Setting | E2 small CNN / E3 ResNet-18 (`configs/small_cnn.yaml`, `configs/resnet18.yaml`) | E1 MedMNIST protocol (`configs/resnet18_medmnist_protocol.yaml`) |
|---|---|---|
| Image size | 28 × 28 × 1 | 28 × 28 × 1 |
| Batch size | 128 | 128 |
| Optimizer | AdamW, weight decay 1e-4 | Adam, no weight decay |
| Learning rate | 1e-3 | 1e-3 |
| Scheduler | cosine over 40 epochs | ×0.1 at epochs 50, 75 |
| Epochs / early stopping | ≤ 40; stop after 10 epochs without val-AUROC gain | 100; no early stop (best val epoch kept) |
| Class weighting | off (E4 ablation: on) | off |
| Augmentation | rotation ≤ 10°, shift ≤ 2 px; no flips (E4 ablation: off) | off |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Checkpoint rule | max val AUROC | max val AUROC |
| Loss | cross-entropy, 2 logits | cross-entropy |

Determinism: `torch.use_deterministic_algorithms(True, warn_only=True)`; a test checks that two CPU runs with the same seed give identical validation scores.

## Class imbalance
Default: no reweighting, with class-wise metrics and a validation-chosen threshold reported. Inverse-frequency class weights are evaluated as ablation E4a, not adopted by default unless validation supports it.

## Resource planning
Measured on this cloud session (4 vCPU Xeon 2.1 GHz, 15 GB RAM, no GPU, ~31 GB free disk), synthetic data of official size (4,708 train images, 28 px), 1 epoch incl. validation:
- Small CNN: ~4.8 s/epoch → 3 seeds × 40 epochs ≈ 10 min per config.
- ResNet-18: ~47.8 s/epoch → 3 seeds × 40 epochs ≈ 1.6 h; E1 (100 epochs × 3 seeds) ≈ 4 h.
Larger sizes (64/128/224 px) are supported by the code but not budgeted for CPU.

## Model card
- Intended use: research/education on benchmarking and explainability; **not** for diagnosis or any clinical decision.
- Data: PneumoniaMNIST (pediatric, single source, 28 px).
- Evaluation: official splits; val for all selection; test once (E7); class-wise metrics, ECE, bootstrap CIs.
- Limitations / failure modes: see README "Limitations". Outputs are not clinical diagnoses.
