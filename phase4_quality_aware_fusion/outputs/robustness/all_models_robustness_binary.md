# All-Models Robustness Report - binary

This report evaluates robustness for Phase 1, Phase 2, Phase 3, and Phase 4 models.

Important:

- Phase 1/2/3 are evaluated with the Phase 3 common-set feature pipeline.
- Phase 4 models are evaluated with the Phase 4 quality-aware feature pipeline.
- Perturbations are applied in memory only. Original dataset files are not modified.

## Clean Consistency Check

| Phase | Model | Clean Acc | Ref Acc | Gap Acc | Clean Macro F1 | Ref Macro F1 | Gap Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | Phase 1 - 2D Common | 93.46% | 93.46% | -0.00 pp | 92.34% | 92.34% | 0.00 pp |
| Phase 2 | Phase 2 - 3D Common | 92.72% | 92.72% | -0.00 pp | 91.40% | 91.40% | 0.00 pp |
| Phase 2 | Phase 2 - 2D+3D Concat Common | 93.93% | 93.93% | -0.00 pp | 92.99% | 92.99% | 0.00 pp |
| Phase 3 | Phase 3 - Gated Fusion | 93.79% | 93.79% | 0.00 pp | 92.81% | 92.81% | -0.00 pp |
| Phase 4 | Phase 4 - Quality-Gated | 95.35% | 95.35% | -0.00 pp | 94.50% | 94.50% | 0.00 pp |
| Phase 4 | Phase 4 - Quality-Concat | 96.16% | 96.16% | 0.00 pp | 95.46% | 95.46% | 0.00 pp |

If the clean gap is large, the robustness comparison must not be used.

## Scenario: clean

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.16% | 95.46% | 0.00 pp | 0.00 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.35% | 94.50% | 0.00 pp | 0.00 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.93% | 92.99% | 0.00 pp | 0.00 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.81% | 0.00 pp | 0.00 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.46% | 92.34% | 0.00 pp | 0.00 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | 0.00 pp |

## Scenario: combined_heavy

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 94.20% | 93.00% | -1.96 pp | -2.46 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 94.03% | 92.81% | -1.32 pp | -1.69 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.74% | 0.00 pp | -0.06 pp |
| 4 | Phase 1 | Phase 1 - 2D Common | 91.57% | 90.35% | -1.89 pp | -2.00 pp |
| 5 | Phase 2 | Phase 2 - 2D+3D Concat Common | 90.69% | 89.58% | -3.24 pp | -3.41 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 91.06% | 89.37% | -1.65 pp | -2.03 pp |

## Scenario: combined_light

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 95.89% | 95.11% | -0.27 pp | -0.36 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.14% | 94.23% | -0.20 pp | -0.27 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 94.03% | 93.06% | 0.24 pp | 0.25 pp |
| 4 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.29% | 92.31% | -0.64 pp | -0.68 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 92.82% | 91.63% | -0.64 pp | -0.71 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.61% | 91.30% | -0.10 pp | -0.10 pp |

## Scenario: frame_drop_0.10

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.12% | 95.42% | -0.03 pp | -0.04 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.38% | 94.55% | 0.03 pp | 0.04 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.96% | 93.03% | 0.03 pp | 0.04 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.66% | 92.66% | -0.13 pp | -0.14 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.39% | 92.25% | -0.07 pp | -0.09 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.65% | 91.33% | -0.07 pp | -0.07 pp |

## Scenario: frame_drop_0.20

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.02% | 95.30% | -0.13 pp | -0.17 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.38% | 94.54% | 0.03 pp | 0.04 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.90% | 92.96% | -0.03 pp | -0.03 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.73% | 92.74% | -0.07 pp | -0.07 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.49% | 92.38% | 0.03 pp | 0.04 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.65% | 91.33% | -0.07 pp | -0.07 pp |

## Scenario: frame_drop_0.30

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.16% | 95.45% | 0.00 pp | -0.01 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.28% | 94.42% | -0.07 pp | -0.08 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.81% | 0.00 pp | 0.00 pp |
| 4 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.73% | 92.75% | -0.20 pp | -0.24 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.56% | 92.46% | 0.10 pp | 0.12 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | -0.01 pp |

## Scenario: gaussian_2d_0.01

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.12% | 95.42% | -0.03 pp | -0.04 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.31% | 94.46% | -0.03 pp | -0.05 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.93% | 92.99% | 0.00 pp | 0.00 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.81% | 0.00 pp | 0.00 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.42% | 92.31% | -0.03 pp | -0.04 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_2d_0.03

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.05% | 95.33% | -0.10 pp | -0.13 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.35% | 94.49% | 0.00 pp | -0.01 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.90% | 92.95% | -0.03 pp | -0.04 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.80% | 0.00 pp | -0.00 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.36% | 92.23% | -0.10 pp | -0.11 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_2d_0.05

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 95.85% | 95.08% | -0.30 pp | -0.38 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.28% | 94.41% | -0.07 pp | -0.10 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.96% | 93.03% | 0.03 pp | 0.04 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.76% | 92.76% | -0.03 pp | -0.05 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.29% | 92.15% | -0.17 pp | -0.19 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | 0.00 pp |

## Scenario: gaussian_3d_0.01

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.16% | 95.46% | 0.00 pp | 0.00 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.35% | 94.50% | 0.00 pp | 0.00 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.93% | 92.99% | 0.00 pp | 0.00 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.76% | 92.76% | -0.03 pp | -0.04 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.46% | 92.34% | 0.00 pp | 0.00 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.65% | 91.32% | -0.07 pp | -0.08 pp |

## Scenario: gaussian_3d_0.03

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.19% | 95.50% | 0.03 pp | 0.04 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.28% | 94.42% | -0.07 pp | -0.08 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 94.00% | 93.07% | 0.07 pp | 0.08 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.83% | 92.84% | 0.03 pp | 0.03 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.46% | 92.34% | 0.00 pp | 0.00 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.68% | 91.37% | -0.03 pp | -0.03 pp |

## Scenario: gaussian_3d_0.05

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.09% | 95.38% | -0.07 pp | -0.09 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.24% | 94.38% | -0.10 pp | -0.12 pp |
| 3 | Phase 2 | Phase 2 - 2D+3D Concat Common | 93.96% | 93.03% | 0.03 pp | 0.04 pp |
| 4 | Phase 3 | Phase 3 - Gated Fusion | 93.79% | 92.80% | 0.00 pp | -0.00 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 93.46% | 92.34% | 0.00 pp | 0.00 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.72% | 91.40% | 0.00 pp | -0.01 pp |

## Scenario: missing_joint_0.10

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 96.05% | 95.31% | -0.10 pp | -0.15 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 95.28% | 94.39% | -0.07 pp | -0.11 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 94.06% | 93.07% | 0.27 pp | 0.26 pp |
| 4 | Phase 2 | Phase 2 - 2D+3D Concat Common | 92.88% | 91.88% | -1.05 pp | -1.11 pp |
| 5 | Phase 1 | Phase 1 - 2D Common | 92.88% | 91.75% | -0.57 pp | -0.60 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 92.11% | 90.68% | -0.61 pp | -0.72 pp |

## Scenario: missing_joint_0.20

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 95.01% | 94.03% | -1.15 pp | -1.43 pp |
| 2 | Phase 3 | Phase 3 - Gated Fusion | 94.47% | 93.53% | 0.67 pp | 0.72 pp |
| 3 | Phase 4 | Phase 4 - Quality-Gated | 94.44% | 93.34% | -0.91 pp | -1.16 pp |
| 4 | Phase 1 | Phase 1 - 2D Common | 91.67% | 90.44% | -1.79 pp | -1.90 pp |
| 5 | Phase 2 | Phase 2 - 2D+3D Concat Common | 91.33% | 90.27% | -2.60 pp | -2.72 pp |
| 6 | Phase 2 | Phase 2 - 3D Common | 91.13% | 89.46% | -1.59 pp | -1.94 pp |

## Scenario: missing_joint_0.30

| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Phase 4 - Quality-Concat | 94.20% | 93.02% | -1.96 pp | -2.44 pp |
| 2 | Phase 4 | Phase 4 - Quality-Gated | 93.93% | 92.65% | -1.42 pp | -1.85 pp |
| 3 | Phase 3 | Phase 3 - Gated Fusion | 93.56% | 92.44% | -0.24 pp | -0.37 pp |
| 4 | Phase 1 | Phase 1 - 2D Common | 89.65% | 88.32% | -3.81 pp | -4.02 pp |
| 5 | Phase 2 | Phase 2 - 3D Common | 89.88% | 87.83% | -2.83 pp | -3.57 pp |
| 6 | Phase 2 | Phase 2 - 2D+3D Concat Common | 83.88% | 82.75% | -10.05 pp | -10.24 pp |

## Interpretation Guide

- Higher Macro F1 under corrupted scenarios means better robustness.
- Smaller negative delta from clean means the model is less sensitive to pose degradation.
- For binary fall detection, the most important comparison is usually Phase 2 Concat vs Phase 4 Quality-Concat.
- For action classification, Phase 2 Concat may still remain the best clean model.
