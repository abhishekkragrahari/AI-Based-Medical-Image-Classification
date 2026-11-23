#!/bin/sh
# Revision R1: ResNet-18 GPU re-runs at 28 and 64 px (seeds 0-2) for the controlled saliency analysis.
set -e
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  $PY -m scripts.train --config configs/revision/resnet18_gpu.yaml --seed $s --out-dir runs/gpu/s28 --run-name resnet18_s$s
  $PY -m scripts.train --config configs/revision/resnet18_gpu.yaml --seed $s --data-path data/pneumoniamnist_64.npz --out-dir runs/gpu/s64 --run-name resnet18_s$s
done
