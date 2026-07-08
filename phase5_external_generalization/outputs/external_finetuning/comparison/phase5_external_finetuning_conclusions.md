# Phase 5 - External Fine-tuning Comparison Report

## 1. Evaluation protocol

In this phase, all model variants were fine-tuned using the same external training split, selected using the same validation split, and finally evaluated on the same external test split. The sequences were prepared in Step 05 and the train/validation/test split was created in Step 06. No sequence rebuilding was performed during model comparison.

This protocol is used to verify whether the improvements introduced in Phase 4 remain effective when the models are adapted to a new dataset.

## 2. Final ranking by external test Macro F1

| Rank | Model | Accuracy | Macro F1 | Fall Recall | Fall F1 | Not_Fall F1 | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | phase4_quality_gated | 75.89% | 75.89% | 75.00% | 75.68% | 76.11% | 43 | 13 | 14 | 42 |
| 2 | phase4_quality_concat | 74.11% | 73.50% | 58.93% | 69.47% | 77.52% | 50 | 6 | 23 | 33 |
| 3 | phase2_concat_common | 72.32% | 72.27% | 67.86% | 71.03% | 73.50% | 43 | 13 | 18 | 38 |
| 4 | phase2_3d_common | 66.96% | 66.36% | 53.57% | 61.86% | 70.87% | 45 | 11 | 26 | 30 |
| 5 | phase3_gated_fusion | 60.71% | 60.70% | 62.50% | 61.40% | 60.00% | 33 | 23 | 21 | 35 |
| 6 | phase1_2d_common | 58.04% | 57.05% | 42.86% | 50.53% | 63.57% | 41 | 15 | 32 | 24 |

## 3. Main findings

The best model by external test Macro F1 is **phase4_quality_gated**, with Macro F1 = **75.89%**, Accuracy = **75.89%**, Fall Recall = **75.00%**, and Fall F1 = **75.68%**.

The best model by Fall Recall is **phase4_quality_gated**, with Fall Recall = **75.00%**.

The best model by Fall F1 is **phase4_quality_gated**, with Fall F1 = **75.68%**.

The two Phase 4 quality-aware models occupy the top two positions by Macro F1. This supports the correctness of the Phase 4 design direction: adding pose-quality information to the fusion process improves external adaptation performance.

Compared with **phase1_2d_common**, **phase4_quality_gated** improves Macro F1 by **18.84 percentage points**, Accuracy by **17.86 points**, Fall Recall by **32.14 points**, and Fall F1 by **25.15 points**.

Compared with **phase2_concat_common**, **phase4_quality_gated** improves Macro F1 by **3.62 percentage points**, Accuracy by **3.57 points**, Fall Recall by **7.14 points**, and Fall F1 by **4.65 points**.

Compared with **phase2_concat_common**, **phase4_quality_concat** improves Macro F1 by **1.23 percentage points**, Accuracy by **1.79 points**, Fall Recall by **-8.93 points**, and Fall F1 by **-1.55 points**.

## 4. Interpretation for the research report

The result shows that the Phase 4 quality-aware fusion strategy remains useful after adapting the models to the external dataset. Although the best internal model in Phase 4 was Quality-Concat, the external adaptation experiment shows that Quality-Gated achieves the strongest overall result on the new test split. Both results still support the same research direction: using pose-quality information improves the fusion process compared with using 2D/3D features alone.

Therefore, the correct conclusion is not that one exact Phase 4 model is always the best in every setting. The stronger conclusion is that the Phase 4 quality-aware design is valid, because the quality-aware models outperform earlier baselines under a fair external fine-tuning protocol.

## 5. Suggested Vietnamese report paragraph

Sau khi toàn bộ mô hình được fine-tune trên cùng tập train của dataset ngoài, chọn checkpoint tốt nhất bằng cùng tập validation và đánh giá trên cùng tập test, hai mô hình thuộc Phase 4 đạt kết quả cao nhất. Cụ thể, Phase 4 Quality-Gated đạt Macro F1 cao nhất trên external test set, đồng thời cũng đạt Fall Recall và Fall F1 tốt nhất. Điều này cho thấy hướng cải tiến ở Phase 4 là có cơ sở: việc đưa đặc trưng chất lượng pose vào quá trình fusion giúp mô hình thích nghi tốt hơn với dữ liệu mới so với các mô hình 2D, 3D, concat thông thường và gated fusion trước đó.
