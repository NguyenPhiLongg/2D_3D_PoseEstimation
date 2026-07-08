# Phase 5 External Fine-tuning Dashboard

## Main conclusion

Best by Macro F1: **phase4_quality_gated** with **75.89%**.
Best by Fall Recall: **phase4_quality_gated** with **75.00%**.
Best by Fall F1: **phase4_quality_gated** with **75.68%**.

After fair external fine-tuning, the Phase 4 quality-aware models achieved the strongest results. This supports the correctness of the Phase 4 quality-aware fusion design.

## Final ranking by test Macro F1

| rank | model_name | best_epoch | test_accuracy | test_macro_f1 | test_fall_recall | test_fall_f1 | test_not_fall_f1 | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | phase4_quality_gated | 10 | 75.89% | 75.89% | 75.00% | 75.68% | 76.11% | 43 | 13 | 14 | 42 |
| 2 | phase4_quality_concat | 8 | 74.11% | 73.50% | 58.93% | 69.47% | 77.52% | 50 | 6 | 23 | 33 |
| 3 | phase2_concat_common | 28 | 72.32% | 72.27% | 67.86% | 71.03% | 73.50% | 43 | 13 | 18 | 38 |
| 4 | phase2_3d_common | 15 | 66.96% | 66.36% | 53.57% | 61.86% | 70.87% | 45 | 11 | 26 | 30 |
| 5 | phase3_gated_fusion | 32 | 60.71% | 60.70% | 62.50% | 61.40% | 60.00% | 33 | 23 | 21 | 35 |
| 6 | phase1_2d_common | 16 | 58.04% | 57.05% | 42.86% | 50.53% | 63.57% | 41 | 15 | 32 | 24 |

## Phase 4 quality-aware models

| model_name | best_epoch | test_accuracy | test_macro_f1 | test_fall_recall | test_fall_f1 | test_not_fall_f1 | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase4_quality_gated | 10 | 75.89% | 75.89% | 75.00% | 75.68% | 76.11% | 43 | 13 | 14 | 42 |
| phase4_quality_concat | 8 | 74.11% | 73.50% | 58.93% | 69.47% | 77.52% | 50 | 6 | 23 | 33 |

## Phase 4 improvement summary

| comparison | model_a | model_b | macro_f1_a | macro_f1_b | macro_f1_gain_points | accuracy_a | accuracy_b | accuracy_gain_points | fall_recall_a | fall_recall_b | fall_recall_gain_points | fall_f1_a | fall_f1_b | fall_f1_gain_points | not_fall_f1_a | not_fall_f1_b | not_fall_f1_gain_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quality-Gated vs 2D baseline | phase4_quality_gated | phase1_2d_common | 0.7589 | 0.5705 | 18.8448 | 0.7589 | 0.5804 | 17.8571 | 0.7500 | 0.4286 | 32.1429 | 0.7568 | 0.5053 | 25.1494 | 0.7611 | 0.6357 | 12.5403 |
| Quality-Gated vs 2D+3D concat | phase4_quality_gated | phase2_concat_common | 0.7589 | 0.7227 | 3.6248 | 0.7589 | 0.7232 | 3.5714 | 0.7500 | 0.6786 | 7.1429 | 0.7568 | 0.7103 | 4.6476 | 0.7611 | 0.7350 | 2.6019 |
| Quality-Gated vs gated fusion | phase4_quality_gated | phase3_gated_fusion | 0.7589 | 0.6070 | 15.1892 | 0.7589 | 0.6071 | 15.1786 | 0.7500 | 0.6250 | 12.5000 | 0.7568 | 0.6140 | 14.2722 | 0.7611 | 0.6000 | 16.1062 |
| Quality-Concat vs 2D baseline | phase4_quality_concat | phase1_2d_common | 0.7350 | 0.5705 | 16.4504 | 0.7411 | 0.5804 | 16.0714 | 0.5893 | 0.4286 | 16.0714 | 0.6947 | 0.5053 | 18.9474 | 0.7752 | 0.6357 | 13.9535 |
| Quality-Concat vs 2D+3D concat | phase4_quality_concat | phase2_concat_common | 0.7350 | 0.7227 | 1.2304 | 0.7411 | 0.7232 | 1.7857 | 0.5893 | 0.6786 | -8.9286 | 0.6947 | 0.7103 | -1.5544 | 0.7752 | 0.7350 | 4.0151 |
| Quality-Concat vs gated fusion | phase4_quality_concat | phase3_gated_fusion | 0.7350 | 0.6070 | 12.7948 | 0.7411 | 0.6071 | 13.3929 | 0.5893 | 0.6250 | -3.5714 | 0.6947 | 0.6140 | 8.0702 | 0.7752 | 0.6000 | 17.5194 |
| Quality-Gated vs Quality-Concat | phase4_quality_gated | phase4_quality_concat | 0.7589 | 0.7350 | 2.3944 | 0.7589 | 0.7411 | 1.7857 | 0.7500 | 0.5893 | 16.0714 | 0.7568 | 0.6947 | 6.2020 | 0.7611 | 0.7752 | -1.4132 |

## Phase-level summary

| phase | num_models | best_macro_model | best_macro_f1 | best_macro_accuracy | best_macro_fall_recall | best_macro_fall_f1 | best_fall_recall_model | best_fall_recall | best_fall_f1_model | best_fall_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1 | 1 | phase1_2d_common | 0.5705 | 0.5804 | 0.4286 | 0.5053 | phase1_2d_common | 0.4286 | phase1_2d_common | 0.5053 |
| Phase 2 | 2 | phase2_concat_common | 0.7227 | 0.7232 | 0.6786 | 0.7103 | phase2_concat_common | 0.6786 | phase2_concat_common | 0.7103 |
| Phase 3 | 1 | phase3_gated_fusion | 0.6070 | 0.6071 | 0.6250 | 0.6140 | phase3_gated_fusion | 0.6250 | phase3_gated_fusion | 0.6140 |
| Phase 4 | 2 | phase4_quality_gated | 0.7589 | 0.7589 | 0.7500 | 0.7568 | phase4_quality_gated | 0.7500 | phase4_quality_gated | 0.7568 |

## Confusion summary

| model_name | display_name | phase | tn | fp | fn | tp | total | false_alarm_rate | miss_rate | fall_detection_rate | not_fall_detection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1_2d_common | Phase 1 - 2D Common | Phase 1 | 41 | 15 | 32 | 24 | 112 | 0.2679 | 0.5714 | 0.4286 | 0.7321 |
| phase2_3d_common | Phase 2 - 3D Common | Phase 2 | 45 | 11 | 26 | 30 | 112 | 0.1964 | 0.4643 | 0.5357 | 0.8036 |
| phase2_concat_common | Phase 2 - 2D+3D Concat | Phase 2 | 43 | 13 | 18 | 38 | 112 | 0.2321 | 0.3214 | 0.6786 | 0.7679 |
| phase3_gated_fusion | Phase 3 - Gated Fusion | Phase 3 | 33 | 23 | 21 | 35 | 112 | 0.4107 | 0.3750 | 0.6250 | 0.5893 |
| phase4_quality_gated | Phase 4 - Quality-Gated | Phase 4 | 43 | 13 | 14 | 42 | 112 | 0.2321 | 0.2500 | 0.7500 | 0.7679 |
| phase4_quality_concat | Phase 4 - Quality-Concat | Phase 4 | 50 | 6 | 23 | 33 | 112 | 0.1071 | 0.4107 | 0.5893 | 0.8929 |

## Report-ready conclusion

After all models were fine-tuned on the same external training split, selected using the same validation split, and evaluated on the same external test split, the Phase 4 models achieved the strongest overall performance. Specifically, Phase 4 Quality-Gated obtained the best Macro F1, Fall Recall, and Fall F1 on the external test set. This result supports the correctness of the Phase 4 improvement direction: incorporating pose-quality features into the fusion process helps the model adapt better to the new dataset compared with the previous 2D, 3D, standard concat, and gated fusion baselines.
