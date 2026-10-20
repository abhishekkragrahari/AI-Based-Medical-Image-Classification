# Research Gap and Contribution Assessment

## Purpose
Identify a defensible research question before implementing a claimed contribution.

## Literature review table
"Verified" = citation details (authors/venue/DOI or arXiv ID) confirmed via web search on 2026-10-18; content summaries are limited to what was confirmed. This is a starting set, not a systematic review.

| # | Citation | Dataset / task | Model / baseline | Split protocol | Metrics | XAI | Relevance | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Yang J. et al. "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification." *Scientific Data* 10, 41 (2023). DOI 10.1038/s41597-022-01721-8; arXiv 2110.14795 | 18 MedMNIST datasets incl. PneumoniaMNIST | ResNet-18/50 at 28 and 224 px, auto-sklearn, AutoKeras, Google AutoML Vision | Official fixed splits; ResNets: Adam lr 1e-3, batch 128, 100 epochs, lr ×0.1 at 50/75, early stopping on val (per arXiv preprint) | AUC, ACC | none | Benchmark + protocol we follow (E1); PneumoniaMNIST ResNet-18 (28) AUC 0.944 / ACC 0.854 and ResNet-18 (224) 0.956 / 0.864 (checked against arXiv 2110.14795, 2026-10-20) | Verified (protocol from arXiv preprint) |
| 2 | Yang J., Shi R., Ni B. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis." ISBI 2021. arXiv 2010.14925 | MedMNIST v1 | as above | official splits | AUC, ACC | none | v1 benchmark; secondary sources quote PneumoniaMNIST ResNet-18 (28) AUC 0.957 / ACC 0.843 from it — **not checked against the primary table** | Citation verified; numbers unverified |
| 3 | Kermany D.S. et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell* 172(5):1122–1131 (2018). DOI 10.1016/j.cell.2018.02.010 | Source pediatric chest X-ray (and OCT) data | transfer learning CNN | source train/test split | — | — | Source dataset of PneumoniaMNIST | Verified |
| 4 | Zech J.R. et al. "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study." *PLOS Medicine* 15(11):e1002683 (2018). DOI 10.1371/journal.pmed.1002683 | 158,323 CXRs from NIH, Mount Sinai, Indiana | CNN | internal vs external sites | AUC | — | Single-site pneumonia models may not generalise; motivates our external-validity limitation | Verified |
| 5 | DeGrave A.J., Janizek J.D., Lee S.-I. "AI for radiographic COVID-19 detection selects shortcuts over signal." *Nature Machine Intelligence* 3:610–619 (2021). DOI 10.1038/s42256-021-00338-7 | COVID-19 CXR datasets | deep CNNs | cross-dataset | AUC | saliency / XAI used to expose confounders | Motivates shortcut (border/marker) checks | Verified |
| 6 | Adebayo J. et al. "Sanity Checks for Saliency Maps." NeurIPS 2018. arXiv 1810.03292 | natural images | — | — | — | some saliency methods are insensitive to model weights/data | Motivates not trusting heatmaps at face value; future: weight-randomisation test | Verified |
| 7 | Petsiuk V., Das A., Saenko K. "RISE: Randomized Input Sampling for Explanation of Black-box Models." BMVC 2018. arXiv 1806.07421 | ImageNet | — | — | deletion / insertion AUC | RISE | Source of the deletion metric we use | Verified |
| 8 | Selvaraju R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017. arXiv 1610.02391 | — | — | — | — | Grad-CAM | Method used | From memory; re-verify |
| 9 | Sundararajan M., Taly A., Yan Q. "Axiomatic Attribution for Deep Networks." ICML 2017. arXiv 1703.01365 | — | — | — | — | Integrated Gradients | Method used | From memory; re-verify |
| 10 | Zeiler M.D., Fergus R. "Visualizing and Understanding Convolutional Networks." ECCV 2014. arXiv 1311.2901 | — | — | — | — | occlusion | Method used | From memory; re-verify |

Literature not yet reviewed: XAI evaluation specifically on MedMNIST/PneumoniaMNIST; calibration studies on MedMNIST; MedMNIST+ (arXiv 2404.15786). Until reviewed, no gap below is claimed to be unaddressed in the literature.

## Candidate gap areas
Treat these as ideas to validate, not established novelty:
- Explainability stability and faithfulness on a low-resolution standard benchmark (chosen focus).
- Calibration of small CNNs vs ResNet-18 on PneumoniaMNIST (secondary, cheap to measure).
- Generalization across institutions — **not testable** here (single source; no external dataset available in this environment).

## Gap validation questions
1. Has the idea already been tested on the same task and datasets? — Unknown; literature search incomplete (see above).
2. Does it improve a meaningful outcome over strong baselines? — This project does not propose a new model; it is an evaluation study.
3. Is the comparison fair and reproducible? — Same official splits, same preprocessing, same tuning budget (none beyond fixed configs), 3 seeds per config.
4. Is the dataset sufficiently independent? — No: single source, no patient IDs; claims limited accordingly.
5. Can improvements be separated from preprocessing/tuning? — Ablations E4 change one factor at a time.
6. Are limitations and negative results reported? — Required in README/PROGRESS.

## Proposed contribution
- Research question: On PneumoniaMNIST (28 px, official splits), how do a small CNN and ResNet-18 compare in discrimination, class-wise error and calibration, and are Grad-CAM, Integrated Gradients and occlusion explanations of these models (a) more faithful than random attributions under a deletion test, (b) stable under small input noise and across training seeds, and (c) free of excess reliance on image borders?
- Hypothesis (H1): ResNet-18 does not significantly outperform the small CNN in val AUROC at 28 px (difference within seed variability). (H2): All three XAI methods yield lower deletion AUC than random attribution. (H3): Cross-seed agreement of explanations is moderate or lower (top-10% IoU < 0.5), i.e. heatmaps are seed-dependent.
- Primary endpoint/metric: validation AUROC (model selection); held-out test AUROC with bootstrap 95% CI (final report). XAI: deletion AUC vs random; top-10% IoU across seeds; border-mass.
- Baselines: small CNN (E2); ResNet-18 (E3); MedMNIST-protocol ResNet-18 (E1).
- Ablations: class weighting on/off; augmentation on/off (small CNN).
- Expected risks: train/val patient leakage inflating val scores; 28 px too coarse for meaningful localisation; small test set (624) → wide CIs; no ground-truth lesion masks, so localisation accuracy cannot be measured.

## Claim policy
Do not claim novelty, clinical benefit, or state-of-the-art performance without adequate evidence and appropriate comparison. This project claims none of these.
