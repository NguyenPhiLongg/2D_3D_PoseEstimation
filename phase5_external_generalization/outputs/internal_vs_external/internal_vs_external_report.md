# Internal vs External Dataset Comparison

## 1. Purpose

This comparison checks whether the model improvement trend from the internal dataset remains valid after adapting the models to a new external dataset. All external results are taken from the fair external fine-tuning protocol: same external train split, same validation split, and same test split for all models.

Important note: internal and external scores should not be interpreted as a direct comparison of dataset difficulty. They are used to compare model ranking and design validity across datasets.

## 2. Model-level comparison

| model_name | phase | internal_macro_rank | external_macro_rank | rank_change | internal_accuracy | external_accuracy | accuracy_change_points | internal_macro_f1 | external_macro_f1 | macro_f1_change_points | external_fall_recall | external_fall_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase4_quality_gated | Phase 4 | 2 | 1 | 1 | 95.35% | 75.89% | -19.45 | 94.50% | 75.89% | -18.61 | 75.00% | 75.68% |
| phase4_quality_concat | Phase 4 | 1 | 2 | -1 | 96.16% | 74.11% | -22.05 | 95.46% | 73.50% | -21.97 | 58.93% | 69.47% |
| phase2_concat_common | Phase 2 | 3 | 3 | 0 | 93.93% | 72.32% | -21.61 | 92.99% | 72.27% | -20.72 | 67.86% | 71.03% |
| phase2_3d_common | Phase 2 | 6 | 4 | 2 | 92.72% | 66.96% | -25.76 | 91.40% | 66.36% | -25.04 | 53.57% | 61.86% |
| phase3_gated_fusion | Phase 3 | 4 | 5 | -1 | 93.79% | 60.71% | -33.08 | 92.81% | 60.70% | -32.11 | 62.50% | 61.40% |
| phase1_2d_common | Phase 1 | 5 | 6 | -1 | 93.46% | 58.04% | -35.42 | 92.34% | 57.05% | -35.29 | 42.86% | 50.53% |

## 3. Phase-level comparison

| phase | num_models | best_internal_model | best_internal_macro_f1 | best_external_model | best_external_macro_f1 | best_external_fall_recall | best_external_fall_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1 | 1 | phase1_2d_common | 92.34% | phase1_2d_common | 57.05% | 42.86% | 50.53% |
| Phase 2 | 2 | phase2_concat_common | 92.99% | phase2_concat_common | 72.27% | 67.86% | 71.03% |
| Phase 3 | 1 | phase3_gated_fusion | 92.81% | phase3_gated_fusion | 60.70% | 62.50% | 61.40% |
| Phase 4 | 2 | phase4_quality_concat | 95.46% | phase4_quality_gated | 75.89% | 75.00% | 75.68% |

## 4. Main conclusion

The exact best model changes from Quality-Concat on the internal dataset to Quality-Gated on the external dataset. However, the top two external models are both Phase 4 quality-aware models. Therefore, the Phase 4 quality-aware fusion direction is supported under external adaptation.

## 5. Caution

Internal and external results should not be interpreted as a direct measure of dataset difficulty because the datasets, splits, and training conditions differ. This comparison is intended to evaluate whether the model ranking trend and design direction remain valid when the models are adapted to a new dataset.

## 6. Report-ready paragraph

The comparison between the internal dataset and the new external dataset shows that the exact best-performing model can change across domains. On the internal dataset, Phase 4 Quality-Concat achieved the best result. After fair fine-tuning on the external dataset, Phase 4 Quality-Gated achieved the best external test result, while Quality-Concat remained among the top-performing models. This suggests that the key contribution is not that one specific architecture always ranks first, but that the Phase 4 quality-aware fusion direction remains valid and beneficial when adapting to a new dataset.
