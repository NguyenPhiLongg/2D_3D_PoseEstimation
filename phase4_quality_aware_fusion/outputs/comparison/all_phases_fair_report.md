# All Phases Fair Comparison Report

This report compares Phase 1, Phase 2, Phase 3, and Phase 4 on the same fair common test sets.

Fair mapping:

- Phase 1 = 2D Common rerun.
- Phase 2 = 3D Common and 2D+3D Concat Common reruns.
- Phase 3 = Gated Fusion on the common set.
- Phase 4 = Quality-Gated and Quality-Concat models.

## Binary Fall Detection

Fair test set: 2965 samples, 812 videos.

| Rank | Phase | Model | Input | Accuracy | Macro F1 | Test Samples | Test Videos |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Phase 4 | Quality-Concat Fusion | 2D pose + 3D pose + quality features | 96.16% | 95.46% | 2965 | 812 |
| 2 | Phase 4 | Quality-Aware Gated Fusion | 2D pose + 3D pose + quality features | 95.35% | 94.50% | 2965 | 812 |
| 3 | Phase 2 | 2D+3D Concat Fusion | 2D pose + estimated 3D pose | 93.93% | 92.99% | 2965 | 812 |
| 4 | Phase 3 | Gated Fusion | 2D pose + estimated 3D pose | 93.79% | 92.81% | 2965 | 812 |
| 5 | Phase 1 | 2D Baseline | 2D pose | 93.46% | 92.34% | 2965 | 812 |
| 6 | Phase 2 | 3D Upgrade | Estimated 3D pose | 92.72% | 91.40% | 2965 | 812 |

Best model for **Binary Fall Detection**: **Phase 4 - Quality-Concat** with **96.16% Accuracy** and **95.46% Macro F1**.

Interpretation: Phase 4 is valuable for binary fall detection. The best binary model is expected to be Quality-Concat, showing that pose quality features help the Fall/Not_Fall decision.

## Action Classification

Fair test set: 2053 samples, 367 videos.

| Rank | Phase | Model | Input | Accuracy | Macro F1 | Test Samples | Test Videos |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Phase 2 | 2D+3D Concat Fusion | 2D pose + estimated 3D pose | 98.64% | 97.51% | 2053 | 367 |
| 2 | Phase 3 | Gated Fusion | 2D pose + estimated 3D pose | 97.81% | 96.18% | 2053 | 367 |
| 3 | Phase 4 | Quality-Aware Gated Fusion | 2D pose + 3D pose + quality features | 97.91% | 95.96% | 2053 | 367 |
| 4 | Phase 4 | Quality-Concat Fusion | 2D pose + 3D pose + quality features | 97.81% | 95.73% | 2053 | 367 |
| 5 | Phase 2 | 3D Upgrade | Estimated 3D pose | 96.79% | 94.70% | 2053 | 367 |
| 6 | Phase 1 | 2D Baseline | 2D pose | 96.98% | 94.36% | 2053 | 367 |

Best model for **Action Classification**: **Phase 2 - Concat Common** with **98.64% Accuracy** and **97.51% Macro F1**.

Interpretation: Phase 3 Concat Fusion can still remain the strongest for multi-class action classification. This means quality features are more useful for fall detection than for fine-grained action recognition.

## Final Task-Specific Recommendation

- Binary Fall Detection: use **Phase 4 - Quality-Concat** (96.16% Accuracy, 95.46% Macro F1).
- Action Classification: use **Phase 2 - Concat Common** (98.64% Accuracy, 97.51% Macro F1).
