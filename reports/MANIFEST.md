# Result manifest

This file maps each table and figure of the manuscript *Benchmarking pneumonia detection and saliency map faithfulness on low-resolution chest X-rays* (Agrahari & Chauhan, revision 1) to the files that hold its numbers.

All paths are relative to the repository root. Unless stated otherwise, results come from GitHub Actions CPU runs. The run IDs are recorded in the files and in `PROGRESS.md`.

Dataset: PneumoniaMNIST v3.0 (Zenodo 10.5281/zenodo.10519652): `pneumoniamnist.npz` (28 px) and `pneumoniamnist_64.npz` (64 px). Integrity is checked in `reports/data_validation_28.json` and `reports/size64/data_validation_64.json`.

## Tables

| Manuscript | Content | Source file(s) |
|---|---|---|
| Table 1 | Split sizes, class counts, duplicates | `reports/data_validation_28.json` |
| Table 2 | Benchmark reproduction and our recipe (test) | Published rows: MedMNIST v2 paper [10]. Benchmark re-run: `reports/e1/test_summary.json`. Our recipe at 28 px: `reports/test_summary.json`. Small CNN at 64 px: `reports/size64/test_summary.json` |
| Section 3.1 | Recipe effect, paired (AUROC +0.022, specificity, balanced accuracy) | `reports/compare_resnet18_recipe_vs_e1.json` |
| Table 3 | Test metrics per configuration at 0.5 | `reports/test_summary.json` (`test_at_0.5`) |
| Section 3.2 | Paired AUROC differences vs ResNet-18; Youden threshold | `reports/test_summary.json` (paired block); `reports/RESULTS.md` |
| Section 3.3 | Validation vs test, duplicate-free validation | `reports/val_summary.json`, `reports/test_summary.json` |
| Table 4 | Small CNN at 28 vs 64 px; paired ensemble differences, including the sensitivity CI | `reports/test_summary.json`, `reports/size64/test_summary.json`, `reports/compare_28_vs_64.json` (sensitivity CI: `scripts/compare_sizes.py`, revision 1) |
| Table 5 | Original single-seed saliency analysis | `reports/xai_resnet18_val/xai_stats.json` (ResNet-18, 28 px), `reports/size64/xai_small_cnn_val/xai_stats.json` (small CNN, 64 px) |
| Table 5b (new) | Controlled saliency analysis: 2 architectures × 2 resolutions × 3 seeds × 2 fills, with paired CIs | `reports/revision/controlled_xai.json` (summary) and `reports/revision/controlled_xai_per_image.json`. GPU re-runs; see below |
| Table 6 | CFR, pre-registered (validation) | `reports/e9/xai_compare_val.json`, `reports/e9/val_small_cnn*.json` |
| Section 3.6 | CFR planned test comparison (Grad-CAM gap −0.0115, p = 0.049) | `reports/e9/xai_compare_test.json` (`groups.cfr.vs_ref`) |
| Table 7 | Comparison with Neo 2026 [20] | Rows for [20]: that paper. Our rows: as Tables 2–4 |

## Figures

| Manuscript | Content | Data source | Plotting script |
|---|---|---|---|
| Figure 1 | Test AUROC per configuration and seed | `reports/test_summary.json`, `reports/e1/test_summary.json`, `reports/size64/test_summary.json` | `scripts/make_paper_figures.py` |
| Figure 2 | Paired AUROC differences | `reports/test_summary.json`, `reports/compare_resnet18_recipe_vs_e1.json`, `reports/compare_28_vs_64.json` | `scripts/make_paper_figures.py` |
| Figure 3 | Confusion matrices of seed ensembles | `reports/test_scores.json`, `reports/size64/test_scores.json` (per-image ensemble probabilities) and official test labels | `scripts/make_paper_figures.py` |
| Figure 4 | Class-wise precision, recall, F1 | `reports/compare_28_vs_64.json` (`per_class`) and `reports/test_scores.json` | `scripts/make_paper_figures.py` |
| Figure 5 | ROC curves of seed ensembles | `reports/test_scores.json`, `reports/size64/test_scores.json`, `reports/e1/test_scores.json` and test labels | `scripts/make_paper_figures.py` |
| Figure 6 | Example explanations, 64 px | `reports/size64/xai_small_cnn_val/examples.png` | `scripts/explain.py` |
| Figure 7 | Deletion AUC vs random (original analysis) | Table 5 sources | `scripts/make_paper_figures.py` |
| Figure 8 | CFR gap differences | `reports/e9/xai_compare_val.json` | `scripts/make_paper_figures.py` |
| Figure 9 | Training loss and validation AUROC per epoch | `reports/training_curves/small_cnn_{28,64}px_s{0,1,2}.json`, `reports/training_curves/val_summary_{28,64}px.json` | `scripts/plot_training_curves.py` |
| New figure | Controlled saliency gaps with CIs | `reports/revision/controlled_xai.json` | `scripts/make_paper_figures.py` |

To regenerate the figures, run `python -m scripts.make_paper_figures`, which writes to `reports/figures/`, and `python -m scripts.plot_training_curves`, which writes Figure 9 to `reports/training_curves/`. The test labels come from the official dataset file, which is not stored in the repository.

## GPU re-runs (revision 1)

The CPU checkpoints behind the classification results were not kept. Figure 9 and the controlled saliency analysis therefore use re-runs with the same configurations and seeds on an NVIDIA RTX 3050 laptop GPU (4 GB). GPU arithmetic is not bit-identical to the CPU runs.

- Small CNN: `configs/small_cnn.yaml` with `device: cuda`, at 28 and 64 px, seeds 0–2.
- ResNet-18: `configs/revision/resnet18_gpu.yaml`, run by `scripts/run_resnet_gpu.sh`.
- Saliency analysis: `scripts/run_controlled_xai.sh` → `scripts/controlled_xai.py`.

These re-runs were used only for explanations and training curves. None of them was evaluated on the test set.
