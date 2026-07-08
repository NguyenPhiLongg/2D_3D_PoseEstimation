# All-Models Robustness Report - action

This report evaluates robustness for Phase 1, Phase 2, Phase 3, and Phase 4 models.

Important:

- Phase 1/2/3 are evaluated with the Phase 3 common-set feature pipeline.
- Phase 4 models are evaluated with the Phase 4 quality-aware feature pipeline.
- Perturbations are applied in memory only. Original dataset files are not modified.

## Clean Consistency Check

| Phase | Model | Clean Acc | Ref Acc | Gap Acc | Clean Macro F1 | Ref Macro F1 | Gap Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | Phase 1 - 2D Common | 96.98% | 96.98% | 0.00 pp | 94.36% | 94.36% | 0.00 pp |
| Phase 2 | Phase 2 - 3D Common | 96.79% | 96.79% | -0.00 pp | 94.70% | 94.70% | 0.00 pp |
| Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 98.64% | -0.00 pp | 97.51% | 97.51% | -0.00 pp |
| Phase 3 | Phase 3 - Gated Fusion | 97.81% | 97.81% | -0.00 pp | 96.18% | 96.18% | 0.00 pp |
| Phase 4 | Phase 4 - Quality-Gated | 97.91% | 97.91% | 0.00 pp | 95.96% | 95.96% | 0.00 pp |
| Phase 4 | Phase 4 - Quality-Concat | 97.81% | 97.81% | -0.00 pp | 95.73% | 95.73% | -0.00 pp |

If the clean gap is large, the robustness comparison must not be used.

## Scenario: clean

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.91% | 95.96% | 0.00 pp | 0.00 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.81% | 95.73% | 0.00 pp | 0.00 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: combined_heavy

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 3 | Phase 3 - Gated Fusion | 89.87% | 82.36% | -7.94 pp | -13.83 pp |
| 2 | Phase 2 | Phase 2 - 2D+3D Concat Common | 89.72% | 80.41% | -8.91 pp | -17.10 pp |
| 3 | Phase 4 | Phase 4 - Quality-Concat | 92.55% | 79.90% | -5.26 pp | -15.83 pp |
| 4 | Phase 1 | Phase 1 - 2D Common | 91.38% | 78.62% | -5.60 pp | -15.74 pp |
| 5 | Phase 4 | Phase 4 - Quality-Gated | 88.85% | 74.85% | -9.06 pp | -21.11 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 80.71% | 74.46% | -16.07 pp | -20.25 pp |

## Scenario: combined_light

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 3 | Phase 3 - Gated Fusion | 97.32% | 95.42% | -0.49 pp | -0.76 pp |
| 2 | Phase 2 | Phase 2 - 2D+3D Concat Common | 97.32% | 94.74% | -1.32 pp | -2.77 pp |
| 3 | Phase 4 | Phase 4 - Quality-Concat | 96.74% | 93.53% | -1.07 pp | -2.20 pp |
| 4 | Phase 2 | Phase 2 - 3D Common | 95.71% | 93.23% | -1.07 pp | -1.47 pp |
| 5 | Phase 4 | Phase 4 - Quality-Gated | 96.15% | 91.78% | -1.75 pp | -4.18 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 95.28% | 89.95% | -1.70 pp | -4.41 pp |

## Scenario: frame_drop_0.10

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.59% | 97.45% | -0.05 pp | -0.06 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.91% | 95.96% | 0.00 pp | 0.00 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.81% | 95.73% | 0.00 pp | 0.00 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 97.17% | 94.73% | 0.19 pp | 0.36 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |

## Scenario: frame_drop_0.20

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.68% | 97.57% | 0.05 pp | 0.06 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.76% | 96.08% | -0.05 pp | -0.10 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.91% | 95.92% | 0.00 pp | -0.05 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.91% | 95.89% | 0.10 pp | 0.16 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 97.03% | 94.39% | 0.05 pp | 0.03 pp |

## Scenario: frame_drop_0.30

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.66% | 95.95% | -0.15 pp | -0.23 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.86% | 95.85% | -0.05 pp | -0.11 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.91% | 95.84% | 0.10 pp | 0.11 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.76% | 0.00 pp | 0.06 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.43% | 0.00 pp | 0.06 pp |

## Scenario: gaussian_2d_0.01

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.95% | 96.06% | 0.05 pp | 0.10 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.81% | 95.68% | 0.00 pp | -0.05 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_2d_0.03

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.91% | 95.99% | 0.00 pp | 0.03 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.76% | 95.56% | -0.05 pp | -0.17 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_2d_0.05

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.71% | 95.51% | -0.19 pp | -0.45 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.71% | 95.38% | -0.10 pp | -0.35 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.79% | 94.70% | 0.00 pp | 0.00 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.35% | 0.00 pp | -0.02 pp |

## Scenario: gaussian_3d_0.01

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.91% | 95.96% | 0.00 pp | 0.00 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.81% | 95.73% | 0.00 pp | 0.00 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.83% | 94.78% | 0.05 pp | 0.08 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_3d_0.03

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 97.95% | 96.09% | 0.05 pp | 0.13 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 97.76% | 96.08% | -0.05 pp | -0.10 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.76% | 95.63% | -0.05 pp | -0.10 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 96.93% | 94.97% | 0.15 pp | 0.27 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_3d_0.05

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 98.64% | 97.51% | 0.00 pp | 0.00 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.81% | 96.18% | 0.00 pp | 0.00 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 97.95% | 96.09% | 0.05 pp | 0.13 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 97.76% | 95.63% | -0.05 pp | -0.10 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 97.03% | 95.06% | 0.24 pp | 0.36 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 96.98% | 94.36% | 0.00 pp | 0.00 pp |

## Scenario: missing_joint_0.10

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | Phase 2 - 2D+3D Concat Common | 97.47% | 95.15% | -1.17 pp | -2.36 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 97.13% | 95.12% | -0.68 pp | -1.06 pp |
| 3 | Phase 2 | Phase 2 - 3D Common | 96.20% | 93.76% | -0.58 pp | -0.94 pp |
| 4 | Phase 4 | Phase 4 - Quality-Concat | 96.74% | 93.34% | -1.07 pp | -2.39 pp |
| 5 | Phase 4 | Phase 4 - Quality-Gated | 96.15% | 91.95% | -1.75 pp | -4.01 pp |
| 6 | Phase 1 | Phase 1 - 2D Common | 95.37% | 89.97% | -1.61 pp | -4.39 pp |

## Scenario: missing_joint_0.20

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 3 | Phase 3 - Gated Fusion | 89.87% | 82.77% | -7.94 pp | -13.41 pp |
| 2 | Phase 2 | Phase 2 - 2D+3D Concat Common | 90.16% | 82.12% | -8.48 pp | -15.38 pp |
| 3 | Phase 4 | Phase 4 - Quality-Concat | 92.74% | 80.87% | -5.07 pp | -14.86 pp |
| 4 | Phase 1 | Phase 1 - 2D Common | 91.96% | 80.66% | -5.02 pp | -13.70 pp |
| 5 | Phase 4 | Phase 4 - Quality-Gated | 91.82% | 78.46% | -6.09 pp | -17.50 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 80.81% | 74.95% | -15.98 pp | -19.75 pp |

## Scenario: missing_joint_0.30

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 1 | Phase 1 - 2D Common | 86.70% | 70.16% | -10.28 pp | -24.21 pp |
| 2 | Phase 4 | Phase 4 - Quality-Concat | 90.21% | 69.43% | -7.60 pp | -26.30 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 85.92% | 68.40% | -11.98 pp | -27.56 pp |
| 4 | Phase 2 | Phase 2 - 2D+3D Concat Common | 75.69% | 53.35% | -22.94 pp | -44.16 pp |
| 5 | Phase 3 | Phase 3 - Gated Fusion | 70.19% | 51.80% | -27.62 pp | -44.38 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 41.11% | 37.63% | -55.67 pp | -57.08 pp |

## Interpretation Guide

- Higher Macro F1 under corrupted scenarios means better robustness.
- Smaller negative delta from clean means the model is less sensitive to pose degradation.
- For binary fall detection, the most important comparison is usually Phase 2 Concat vs Phase 4 Quality-Concat.
- For action classification, Phase 2 Concat may still remain the best clean model.
