# 2D/3D Human Pose Fall Detection and Quality-Aware Fusion

This repository contains a research implementation for **fall detection** and **human action recognition** using 2D pose estimation, estimated 3D pose lifting, temporal sequence modeling, and multiple fusion strategies.

The project compares several model families:

- **Phase 1:** 2D pose baseline
- **Phase 2:** estimated 3D pose and 2D+3D concatenation
- **Phase 3:** gated 2D/3D fusion
- **Phase 4:** quality-aware fusion
- **Phase 5:** external dataset adaptation and validation

The final research direction shows that **quality-aware fusion** improves robustness when adapting to a new external dataset.

---

## 1. Project Overview

The system takes video clips as input and predicts whether the action is a fall or a normal activity.

### Main Tasks

#### Binary Fall Detection

```text
Fall
Not_Fall
```

#### Action Classification

```text
Sitting
Sleeping
Standing
Walking
```

### General Pipeline

```text
Raw video
   ↓
YOLOv8-Pose
   ↓
2D pose keypoints
   ↓
PoseFormerV2 / 2D-to-3D pose lifting
   ↓
Estimated 3D pose features
   ↓
2D / 3D / fusion sequence models
   ↓
Fall detection or action classification
```

---

## 2. Research Motivation

Fall detection is important in healthcare monitoring, elderly care, and intelligent surveillance. Missing a real fall can be dangerous, while false alarms can reduce system reliability.

Instead of relying directly on RGB frames, this project uses skeleton-based representations. Skeleton features are more compact and focus on human body structure rather than background, lighting, or irrelevant image details.

The main research questions are:

1. Can estimated 3D pose improve fall detection compared with 2D pose only?
2. Can 2D and 3D pose features be fused effectively?
3. Can pose-quality information improve fusion robustness?
4. Do the improvements remain valid on an external dataset?

---

## 3. Dataset Sources

Raw datasets are not included in this repository because of file size limitations.

### 3.1 Internal Dataset

The internal experiments use fall and non-fall videos collected from public video datasets and organized into fall and action classes.

The main internal sources include:

- Fall Vision dataset
- Human Activity & Suspicious Behavior Video Dataset from Kaggle
- Additional curated class folders for sitting, sleeping, standing, and walking

### 3.2 External Dataset for Phase 5

Phase 5 uses an external dataset adaptation protocol. The main external dataset used for binary adaptation is **MulCamFall**.

Final valid external dataset split:

| Split | Sequences | Videos | Groups / Chutes |
|---|---:|---:|---:|
| Train | 335 | 120 | 15 |
| Validation | 104 | 32 | 4 |
| Test | 112 | 32 | 4 |
| Total | 551 | 184 | 23 |

The external test split is balanced:

```text
Not_Fall: 56 sequences
Fall: 56 sequences
```

---

## 4. Feature Representations

### 4.1 2D Features

The 2D model uses 17 COCO keypoints.

Each keypoint has:

```text
x, y
```

The final 2D frame representation is:

```text
34 normalized 2D keypoint values
+ 6 handcrafted 2D pose features
= 40 features per frame
```

Example handcrafted 2D features:

- aspect ratio
- normalized width
- normalized height
- normalized center position
- motion velocity

### 4.2 3D Features

The estimated 3D model uses 17 skeleton joints.

Each joint has:

```text
x, y, z
```

The final 3D frame representation is:

```text
51 normalized 3D coordinate values
+ 8 handcrafted 3D pose features
= 59 features per frame
```

Example handcrafted 3D features:

- width
- depth
- height
- height-width ratio
- depth-width ratio
- head height
- torso tilt
- velocity

### 4.3 Concatenated 2D+3D Features

The early fusion representation concatenates 2D and 3D features:

```text
2D features + 3D features
= 40 + 59
= 99 features per frame
```

### 4.4 Pose Quality Features

Phase 4 introduces quality-aware fusion. Quality features describe the reliability of pose extraction and 3D lifting.

Examples:

- mean keypoint confidence
- missing joint ratio
- low-confidence keypoint ratio
- multi-person ratio
- bounding box stability
- temporal jitter
- 2D/3D motion instability
- 3D depth instability

These quality features help the fusion model decide how much it should trust 2D and 3D pose information.

---

## 5. Model Variants

| Phase | Model | Description |
|---|---|---|
| Phase 1 | `phase1_2d_common` | 2D pose sequence baseline |
| Phase 2 | `phase2_3d_common` | estimated 3D pose sequence model |
| Phase 2 | `phase2_concat_common` | early concatenation of 2D and 3D features |
| Phase 3 | `phase3_gated_fusion` | learned gated fusion between 2D and 3D streams |
| Phase 4 | `phase4_quality_gated` | quality-aware gated fusion |
| Phase 4 | `phase4_quality_concat` | 2D+3D concat with additional quality features |

Most sequence models use a CNN1D + BiLSTM style architecture for temporal modeling.

Input shape:

```text
(batch_size, sequence_length, feature_dim)
```

Default sequence configuration:

```text
sequence_length = 60
stride = 15
```

---

## 6. Project Structure

A simplified project structure is shown below.

```text
3D_Human_Pose_NCKH/
├── data/
│   ├── 1_raw_videos/
│   ├── 2_extracted_2d/
│   ├── 3_extracted_3d/
│   ├── 4_normalized_3d/
│   └── master_dataset.csv
│
├── phase1_2d_baseline/
│   ├── 01_yolo_extract.py
│   ├── model_2d.py
│   ├── train_2d.py
│   ├── checkpoints/
│   └── outputs/
│
├── phase2_3d_upgrade/
│   ├── adapters/
│   ├── inference/
│   ├── notebooks/
│   ├── model_3d.py
│   ├── extract_3d_keypoints.py
│   ├── normalize_3d_dataset.py
│   ├── train_3d.py
│   ├── train_fusion_2d3d.py
│   ├── checkpoints/
│   └── outputs/
│
├── phase3_common_set_gated_fusion/
│   ├── model_gated_fusion.py
│   ├── checkpoints/
│   └── outputs/
│
├── phase4_quality_aware_fusion/
│   ├── model_quality_aware_fusion.py
│   ├── checkpoints/
│   └── outputs/
│
├── phase5_external_generalization/
│   ├── phase5_config.yaml
│   ├── phase5_utils.py
│   ├── external_label_mapping.py
│   ├── scripts/
│   │   ├── 01_prepare_external_dataset.py
│   │   ├── 02_extract_2d_confidence_external.py
│   │   ├── 03_estimate_3d_external.py
│   │   ├── 04_build_external_quality_features.py
│   │   ├── 05_build_external_sequences.py
│   │   ├── 06_create_external_train_val_test_split.py
│   │   ├── 07_finetune_all_models_external.py
│   │   ├── 08_compare_external_finetuning_results.py
│   │   ├── 09_generate_phase5_dashboard.py
│   │   └── 10_compare_internal_vs_external.py
│   ├── data/
│   └── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

Large data folders, raw videos, generated NumPy files, and model checkpoints are excluded from Git by default.

---

## 7. Installation

### 7.1 Create Conda Environment

```bash
conda create -n nckh_3dpose python=3.10 -y
conda activate nckh_3dpose
```

### 7.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Install the correct PyTorch version for your CUDA version from the official PyTorch installation page.

### 7.3 Optional: Check CUDA

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## 8. How to Run

### 8.1 Phase 1: 2D Baseline

Extract 2D keypoints:

```bash
python phase1_2d_baseline/01_yolo_extract.py
```

Train 2D binary model:

```bash
python phase1_2d_baseline/train_2d.py --task binary --epochs 30
```

Train 2D action model:

```bash
python phase1_2d_baseline/train_2d.py --task action --epochs 30
```

### 8.2 Phase 2: 3D Upgrade and 2D+3D Concatenation

Extract estimated 3D keypoints:

```bash
python phase2_3d_upgrade/extract_3d_keypoints.py
```

Normalize 3D keypoints:

```bash
python phase2_3d_upgrade/normalize_3d_dataset.py --overwrite
```

Train 3D model:

```bash
python phase2_3d_upgrade/train_3d.py --task binary --epochs 30
python phase2_3d_upgrade/train_3d.py --task action --epochs 30
```

Train early fusion 2D+3D model:

```bash
python phase2_3d_upgrade/train_fusion_2d3d.py --task binary --epochs 30
python phase2_3d_upgrade/train_fusion_2d3d.py --task action --epochs 30
```

### 8.3 Phase 5: External Dataset Adaptation

Run the full Phase 5 pipeline:

```bash
python phase5_external_generalization/scripts/01_prepare_external_dataset.py
python phase5_external_generalization/scripts/02_extract_2d_confidence_external.py
python phase5_external_generalization/scripts/03_estimate_3d_external.py
python phase5_external_generalization/scripts/04_build_external_quality_features.py
python phase5_external_generalization/scripts/05_build_external_sequences.py
python phase5_external_generalization/scripts/06_create_external_train_val_test_split.py
python phase5_external_generalization/scripts/07_finetune_all_models_external.py
python phase5_external_generalization/scripts/08_compare_external_finetuning_results.py
python phase5_external_generalization/scripts/09_generate_phase5_dashboard.py
python phase5_external_generalization/scripts/10_compare_internal_vs_external.py
```

Fast test for one model:

```bash
python phase5_external_generalization/scripts/07_finetune_all_models_external.py --models phase1_2d_common --epochs 2
```

Run full external fine-tuning:

```bash
python phase5_external_generalization/scripts/07_finetune_all_models_external.py --epochs 40 --patience 10 --batch-size 32
```

---

## 9. Experimental Results

### 9.1 Internal Dataset Results

| Model | Accuracy | Macro F1 | Fall Recall | Fall F1 |
|---|---:|---:|---:|---:|
| Phase 1 - 2D Common | 93.46% | 92.34% | 89.91% | 89.42% |
| Phase 2 - 3D Common | 92.72% | 91.40% | 87.17% | 88.04% |
| Phase 2 - 2D+3D Concat | 93.93% | 92.99% | 93.20% | 90.43% |
| Phase 3 - Gated Fusion | 93.79% | 92.81% | 92.21% | 90.14% |
| Phase 4 - Quality-Gated | 95.35% | 94.50% | N/A | N/A |
| Phase 4 - Quality-Concat | 96.16% | 95.46% | N/A | N/A |

On the internal dataset, **Phase 4 Quality-Concat** achieves the strongest overall binary result.

### 9.2 External Dataset Fine-tuning Results

All models are fine-tuned on the same external train split, selected using the same validation split, and evaluated on the same external test split.

| Rank | Model | Best Epoch | Accuracy | Macro F1 | Fall Recall | Fall F1 | Not_Fall F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Phase 4 - Quality-Gated | 10 | 75.89% | 75.89% | 75.00% | 75.68% | 76.11% |
| 2 | Phase 4 - Quality-Concat | 8 | 74.11% | 73.50% | 58.93% | 69.47% | 77.52% |
| 3 | Phase 2 - 2D+3D Concat | 28 | 72.32% | 72.27% | 67.86% | 71.03% | 73.50% |
| 4 | Phase 2 - 3D Common | 15 | 66.96% | 66.36% | 53.57% | 61.86% | 70.87% |
| 5 | Phase 3 - Gated Fusion | 32 | 60.71% | 60.70% | 62.50% | 61.40% | 60.00% |
| 6 | Phase 1 - 2D Common | 16 | 58.04% | 57.05% | 42.86% | 50.53% | 63.57% |

The best external model is **Phase 4 Quality-Gated**.

### 9.3 Main Research Finding

The exact best-performing model changes across datasets:

- Internal dataset: **Quality-Concat** performs best.
- External dataset: **Quality-Gated** performs best.

However, the top external models are still both **Phase 4 quality-aware models**. This supports the main research conclusion that **adding pose-quality information improves fusion robustness and helps the model adapt better to external data**.

---

## 10. Important Outputs

### Phase 5 External Fine-tuning

```text
phase5_external_generalization/outputs/external_finetuning/
```

Important files:

```text
external_finetuned_all_models_metrics.csv
external_finetuned_all_models_per_epoch.csv
```

### Phase 5 Comparison Report

```text
phase5_external_generalization/outputs/external_finetuning/comparison/
```

Important files:

```text
ranking_by_macro_f1.csv
phase4_improvement_summary.csv
phase5_external_finetuning_conclusions.md
08_compare_external_finetuning_results_report.json
```

### Phase 5 Dashboard

```text
phase5_external_generalization/outputs/phase5_dashboard/
```

Important files:

```text
phase5_dashboard.html
phase5_dashboard.md
phase5_dashboard_summary.json
```

### Internal vs External Comparison

```text
phase5_external_generalization/outputs/internal_vs_external/
```

Important files:

```text
internal_vs_external_report.html
internal_vs_external_report.md
internal_vs_external_model_comparison.csv
10_compare_internal_vs_external_report.json
```

---

## 11. Reproducibility Notes

Default configuration:

```text
random_seed = 42
sequence_length = 60
stride = 15
```

External fine-tuning protocol:

```text
same external train split
same external validation split
same external test split
best checkpoint selected by validation Macro F1
final ranking based on test Macro F1
```

This avoids unfair comparison caused by different train/test splits.

---

## 12. Important Limitations

### 12.1 Estimated 3D Is Not Ground-truth 3D

The 3D skeletons are estimated from 2D keypoints. They are not captured from motion capture systems.

Errors may come from:

- 2D pose detection
- missing keypoints
- camera angle
- 2D-to-3D lifting
- temporal instability

Therefore, 3D-only models do not always outperform 2D models.

### 12.2 Dataset Shift

External datasets may have different:

- camera viewpoints
- environments
- fall definitions
- action distributions
- video quality
- pose detection reliability

For this reason, external evaluation should be interpreted as a domain adaptation test, not as a direct comparison of dataset difficulty.

---

## 13. Repository and Data Notes

The following files are intentionally excluded from Git:

```text
raw videos
extracted 2D keypoint CSV files
estimated 3D CSV files
large NumPy arrays
model checkpoints
pretrained weights
temporary logs
large compressed archives
```

If large files need to be shared, use:

- Google Drive
- Kaggle
- GitHub Releases
- Git LFS
- institutional storage

---

## 14. Future Work

Possible improvements:

- Add more balanced fall and non-fall videos.
- Improve 2D pose quality filtering.
- Fine-tune the 3D pose lifting model on fall-related poses.
- Test temporal models such as TCN, Transformer, or ST-GCN.
- Improve domain adaptation for external datasets.
- Add real-time webcam or CCTV inference.
- Use temporal smoothing to reduce unstable frame-level predictions.
- Evaluate on more camera angles and environments.

---

## 15. Research Conclusion

This project demonstrates a complete skeleton-based fall detection pipeline using 2D pose, estimated 3D pose, fusion models, and external dataset validation.

The main conclusion is:

```text
Quality-aware fusion is more robust than using 2D pose, 3D pose, or simple 2D+3D concatenation alone when adapting to an external dataset.
```

The Phase 5 experiment supports the validity of the Phase 4 quality-aware fusion design.
