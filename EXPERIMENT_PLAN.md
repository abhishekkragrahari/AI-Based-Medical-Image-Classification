# Experiment Plan

## Before experiments
Freeze the research question, dataset version, split protocol, primary metric, baselines, and analysis plan.

## Experiment matrix
| ID | Experiment | Purpose | Status |
|---|---|---|---|
| E0 | Data validation and leakage audit | Establish data integrity | Planned |
| E1 | Reproduce repository baseline | Verify starting point | Planned |
| E2 | Train simple CNN baseline | Establish reference performance | Planned |
| E3 | Compare selected pretrained model(s) | Controlled model comparison | Planned |
| E4 | Ablation of proposed component | Test contribution | Planned |
| E5 | Explainability evaluation | Inspect explanation quality/stability | Planned |
| E6 | Robustness or cross-domain evaluation | Test generalization where data permits | Planned |
| E7 | Final held-out evaluation | Final unbiased assessment | Planned |

## Metrics
Select metrics based on the task and class distribution. Possible choices include accuracy, balanced accuracy, precision, recall/sensitivity, specificity, macro-F1, AUROC, AUPRC, confusion matrix, calibration error, and confidence intervals. Not all metrics suit every task.

## Fair comparison
- Use identical splits and consistent preprocessing.
- Document tuning budgets and compute.
- Tune only on training/validation data.
- Use multiple seeds when feasible.
- Report mean and variability when supported.
- Do not compare results from incompatible test sets as if directly comparable.

## Results table
| Run ID | Commit | Dataset/split | Model/config | Seed | Metrics artifact | Notes |
|---|---|---|---|---|---|---|
| [fill in] | [fill in] | [fill in] | [fill in] | [fill in] | [path] | [fill in] |

## Reporting
Include negative results, errors, confidence/uncertainty where feasible, and limitations. Never invent missing values.
