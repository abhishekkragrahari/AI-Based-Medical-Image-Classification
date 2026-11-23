#!/bin/sh
# Revision R1: controlled saliency analysis, 2 architectures x 2 resolutions x 3 seeds (GPU re-runs).
set -e
PY=.venv/Scripts/python.exe
ck() { echo "runs/gpu/$1/$2_s0/best.pt,runs/gpu/$1/$2_s1/best.pt,runs/gpu/$1/$2_s2/best.pt"; }
$PY -m scripts.controlled_xai --device cuda \
  --config "cnn28=$(ck s28 small_cnn)@data/pneumoniamnist.npz" \
           "cnn64=$(ck s64 small_cnn)@data/pneumoniamnist_64.npz" \
           "res28=$(ck s28 resnet18)@data/pneumoniamnist.npz" \
           "res64=$(ck s64 resnet18)@data/pneumoniamnist_64.npz" \
  --out reports/revision/controlled_xai.json
