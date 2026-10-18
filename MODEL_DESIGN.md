# Model Design

## Design principle
Start with a simple, reproducible baseline. Add complexity only when it tests a defined hypothesis.

## Candidate models
- Small CNN: transparent initial baseline.
- ResNet family: established convolutional baseline.
- EfficientNet family: compute-efficient image model.
- Vision Transformer: candidate when data and compute justify it.

Choose models based on task, dataset size, pretrained-weight terms, and available hardware.

## Pipeline
1. Input validation.
2. Dataset loading and transformations.
3. Model initialization.
4. Training and validation.
5. Checkpoint selection using validation data only.
6. One final evaluation on the held-out test set.
7. Prediction and explanation generation.

## Training configuration
Record image size, batch size, optimizer, learning rate, scheduler, epochs, early-stopping rule, class weighting, augmentation, random seeds, and checkpoint selection rule.

## Class imbalance
Consider class weights, balanced sampling, or suitable losses only when justified. Report per-class metrics and the exact approach.

## Resource planning
Check CPU, RAM, GPU, VRAM, and disk before training. Start with a small smoke test and tune batch size to available memory. Keep a CPU path where feasible.

## Model card
Document intended use, data, evaluation, limitations, failure modes, and the fact that outputs are not clinical diagnoses.
