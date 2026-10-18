# Explainable AI (XAI) Plan

## Objective
Inspect what evidence a trained model uses and whether explanations are stable and meaningful. Visual explanations do not establish clinical correctness.

## Candidate methods
- Grad-CAM or related gradient-based maps for suitable convolutional models.
- Integrated Gradients where model and input format support it.
- Occlusion sensitivity as a model-agnostic comparison.
- Attention visualization only when its interpretation is justified; attention alone is not proof of explanation.

## Procedure
1. Select checkpoints using validation data only.
2. Generate explanations for correct and incorrect predictions.
3. Include different classes, confidence levels, and difficult cases.
4. Use consistent visualization scales and document settings.
5. Compare explanations across seeds or perturbations where feasible.
6. Test for reliance on borders, markers, text, or other shortcuts.

## Evaluation
Where reference regions or annotations exist, choose suitable localization metrics. Otherwise, assess stability, sensitivity to perturbations, and known failure modes. Avoid unsupported claims that heatmaps are faithful or clinically valid.

## Deliverables
- Saved overlays with sample identifiers that do not reveal private information.
- Scripts and configurations to regenerate explanations.
- A written analysis of representative successes and failures.
