# Literature and Novelty Check: XAI Faithfulness vs Image Resolution

Date: 2026-10-24. Scope: targeted web search (not a systematic review). Question: is "explanation faithfulness depends on input resolution (MedMNIST / chest X-ray)" new enough for a paper?

"Read" = full text or full abstract page read; "Abstract-level" = only search snippets/abstract seen. Re-verify every entry before citing.

## Closest prior work

| # | Citation | What it does | Overlap with this project | Status |
|---|---|---|---|---|
| 11 | Kulkarni A.V., Zech J.R., Yi P.H. "Sharpness matters: Higher image resolution improves generalization and explainability in chest X-ray classification." *Scientific Reports* 16, 21856 (2026). DOI 10.1038/s41598-026-49129-3 | SIIM-ACR pneumothorax + RSNA pneumonia, patient-level splits, external test sets; DenseNet-121/ResNet-152 (ImageNet-pretrained) at 64–1024 px; **Grad-CAM only**; explainability = overlap with ground-truth masks/boxes (precision) and map coverage | Same "resolution vs explainability in CXR" theme. **Differs:** measures localisation, not faithfulness (no deletion/insertion, no randomisation checks); lowest resolution 64 px; one XAI method; no cross-seed agreement | Read (PMC full text) |
| 12 | "TRAX: A framework for quantifying trustworthiness of explainable AI in medical image classification." ScienceDirect, journal S2590005626005576 (2026). Authors/volume not yet recorded | 5 MedMNIST datasets incl. **PneumoniaMNIST**; 5 CNNs; 375 configurations across "dataset-quality variants" and seeds; stability, faithfulness (deletion), robustness, consistency combined into a trust score; bootstrap CIs. Reports weak correlation between accuracy and explanation trust (ρ = 0.082) | **Largest overlap**: same benchmark, same kind of metrics, seeds and CIs. Unknown whether "dataset-quality variants" include resolution, and whether a random-attribution baseline is used | Abstract-level only (full text returned HTTP 403). **Must read before writing** |
| 13 | Doerrich S. et al. "Rethinking model prototyping through the MedMNIST+ dataset collection." *Scientific Reports* (2025). DOI 10.1038/s41598-025-92156-9; arXiv 2404.15786 | MedMNIST+ at 28/64/128/224 px; CNNs vs ViTs; higher resolution does not consistently improve accuracy beyond a threshold | Resolution vs **accuracy** only; no XAI evaluation found | Abstract-level |
| 14 | Arun N. et al. "Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging." *Radiology: AI* (2021). DOI 10.1148/ryai.2021200267 | 8 saliency methods on CXR; randomisation, repeatability, reproducibility, localisation | Saliency reliability in CXR; no resolution study | Abstract-level |
| 15 | Saporta A. et al. "Benchmarking saliency methods for chest X-ray interpretation." *Nature Machine Intelligence* (2022); medRxiv 10.1101/2021.02.28.21252634 | 7 methods vs radiologist localisation on CheXpert (320 px); Grad-CAM best but below humans; worst for small pathologies | Localisation, fixed resolution | Abstract-level |
| 16 | "Back to the Baseline: Examining Baseline Effects on Explainability Metrics." arXiv 2512.11433 | Deletion/insertion results and method rankings change strongly with the replacement baseline | **Methodological threat**: at 28 px a 4×4 patch or 0-fill is a large, out-of-distribution change, so a "resolution effect" could be an artefact of the metric | Abstract-level |
| 17 | AttributionLab (arXiv 2310.06514) | Perturbation faithfulness scores mislead when perturbed inputs are unseen by the model ("unseen data effect") | Same threat as #16 | Abstract-level |

Also found: public GitHub projects doing similar PneumoniaMNIST XAI checks (`Gaova777/whitebox-medmnist`, 128 px, stability; `PrithS24/pneumonia-cxr-resnet18-explainability`, 224 px, Grad-CAM vs occlusion, randomisation). Not peer-reviewed, but they show the basic idea is common.

## Verdict
- **As the results stand now (1 dataset, 28 vs 64 px, 3 seeds): not new enough.** TRAX already covers faithfulness, stability and seeds on PneumoniaMNIST with CIs; Kulkarni et al. already show explanations improve with resolution in CXR. A reviewer would see this as a smaller version of both.
- **Remaining gap (hypothesis, pending the TRAX full text):** no paper found that measures **perturbation faithfulness against a random-attribution baseline as a function of input resolution (28→224 px)**, for several XAI methods, while **controlling for the metric's own resolution dependence** (fill value, patch size relative to image, insertion vs deletion). Kulkarni measure localisation; MedMNIST+ measures accuracy; TRAX's resolution coverage is unknown.
- If TRAX turns out to vary resolution, the gap mostly closes; the work would then fit only as a short replication/negative-results paper.

## What would make it publishable
1. Read TRAX in full and position against it and Kulkarni et al. explicitly.
2. Resolutions 28/64/128/224 (MedMNIST+), not two points.
3. At least 3 MedMNIST datasets (e.g. Pneumonia, Breast, Derma or Blood).
4. Metric controls: several fill baselines (0, blur, train mean), deletion **and** insertion, patch size scaled to image size, and an Adebayo model-randomisation sanity check — so a resolution effect cannot be explained by the metric alone.
5. Fresh test usage plan: the PneumoniaMNIST test set has already been looked at several times; pre-register and use val for the XAI analysis, test only once for accuracy.
6. Realistic venue: workshop (e.g. iMIMIC / MICCAI workshops, MIDL short paper) or an open-access journal; not a top-tier venue.
