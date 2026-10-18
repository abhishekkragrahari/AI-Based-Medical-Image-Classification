# Explainable AI (XAI) Plan

## Objective
Inspect what evidence a trained model uses and whether explanations are stable and meaningful. Visual explanations do not establish clinical correctness.

## Methods (implemented in `medxai/xai.py`)
| Method | Settings | Notes |
|---|---|---|
| Grad-CAM (Selvaraju et al., 2017) | last conv block (`SmallCNN.block3`: 7×7; `ResNet18Small.layer4`: 4×4 at 28 px), ReLU, bilinear upsampling to 28×28 | coarse at 28 px |
| Integrated Gradients (Sundararajan et al., 2017) | baseline = 0 in normalised space (= training mean intensity), 32 midpoint steps, |attribution| summed over channel | baseline choice affects maps |
| Occlusion (Zeiler & Fergus, 2014) | 4×4 patch, stride 2, fill = training mean; drop in predicted-class probability, clipped at 0 | model-agnostic reference |

All maps explain the **predicted** class and are min–max normalised per image to [0, 1]; overlays use the same `jet` colormap and α = 0.45.
Attention visualisation: not applicable (no attention models).

## Procedure
1. Checkpoints selected on validation data only (best val AUROC epoch).
2. `scripts/explain.py` picks up to N examples each of TP, TN, FP, FN, spread evenly over confidence |p − 0.5|.
3. Different classes, confidence levels and errors are therefore included by construction.
4. Consistent scales/settings as above; recorded in `xai_stats.json`.
5. Cross-seed comparison: explanations from seeds 1–2 vs seed 0 for the same image and target class (top-10% IoU); noise stability: Spearman between maps for x and x + N(0, 0.05²) (normalised units), 3 draws, 30 images.
6. Shortcut test: border mass = share of attribution in the outer 10% frame (3 px at 28 px). Uniform reference = 0.383 at 28 px.

E5 runs on the **validation** split. Test-split explanations require `--final` and are produced only after E7.

## Evaluation
No lesion masks exist for PneumoniaMNIST, so localisation accuracy cannot be measured. We report:
- Faithfulness: deletion AUC (Petsiuk et al., 2018) per method vs random-order deletion on the same images (lower = more faithful).
- Agreement between methods (Spearman).
- Stability under noise and across seeds.
- Border-mass vs uniform reference.
Not done yet (future): model-parameter randomisation sanity check (Adebayo et al., 2018).

## Deliverables
- `reports/xai_<model>_val/examples.png` — overlays labelled with split index only (no patient information exists).
- `reports/xai_<model>_val/xai_stats.json` — all statistics above.
- Written analysis — to be added to README/PROGRESS after E5 runs on real data. **Status: blocked (data download).**
