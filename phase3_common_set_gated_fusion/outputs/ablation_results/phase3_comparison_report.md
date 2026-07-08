# Phase 3 Result Comparison

## Purpose

Phase 3 evaluates 2D-only, 3D-only, concat fusion, and gated fusion on the same common video subset. This ensures that the comparison is fair.

## Binary Task: Fall / Not_Fall

          experiment input input_dim  accuracy  macro_f1  fall_precision  fall_recall  fall_f1  num_test_samples  num_test_unique_videos  mean_gate
           2D Common    2D       40D    0.9346    0.9234          0.8894       0.8991   0.8942              2965                     812        NaN
           3D Common    3D       59D    0.9272    0.9140          0.8893       0.8717   0.8804              2965                     812        NaN
Concat Fusion Common 2D+3D       99D    0.9393    0.9299          0.8781       0.9320   0.9043              2965                     812        NaN
        Gated Fusion 2D+3D   40D+59D    0.9379    0.9281          0.8816       0.9221   0.9014              2965                     812     0.4837

## Action Task: Sitting / Sleeping / Standing / Walking

          experiment input input_dim  accuracy  macro_f1  sitting_f1  sleeping_f1  standing_f1  walking_f1  num_test_samples  num_test_unique_videos  mean_gate
           2D Common    2D       40D    0.9698    0.9436      0.9404       0.9639       0.8730      0.9973              2053                     367        NaN
           3D Common    3D       59D    0.9679    0.9470      0.9521       0.9488       0.8963      0.9909              2053                     367        NaN
Concat Fusion Common 2D+3D       99D    0.9864    0.9751      0.9741       0.9770       0.9497      0.9995              2053                     367        NaN
        Gated Fusion 2D+3D   40D+59D    0.9781    0.9618      0.9599       0.9575       0.9313      0.9986              2053                     367     0.4663

## Improvement Analysis

  task         target_model       baseline_model   metric  target_value  baseline_value  absolute_improvement  relative_improvement_percent
binary Concat Fusion Common            2D Common accuracy        0.9393          0.9346                0.0047                        0.5029
binary Concat Fusion Common            2D Common macro_f1        0.9299          0.9234                0.0065                        0.7039
binary Concat Fusion Common            3D Common accuracy        0.9393          0.9272                0.0121                        1.3050
binary Concat Fusion Common            3D Common macro_f1        0.9299          0.9140                0.0159                        1.7396
binary         Gated Fusion Concat Fusion Common accuracy        0.9379          0.9393               -0.0014                       -0.1490
binary         Gated Fusion Concat Fusion Common macro_f1        0.9281          0.9299               -0.0018                       -0.1936
binary         Gated Fusion            2D Common accuracy        0.9379          0.9346                0.0033                        0.3531
binary         Gated Fusion            2D Common macro_f1        0.9281          0.9234                0.0047                        0.5090
binary         Gated Fusion            3D Common accuracy        0.9379          0.9272                0.0107                        1.1540
binary         Gated Fusion            3D Common macro_f1        0.9281          0.9140                0.0141                        1.5427
action Concat Fusion Common            2D Common accuracy        0.9864          0.9698                0.0166                        1.7117
action Concat Fusion Common            2D Common macro_f1        0.9751          0.9436                0.0315                        3.3383
action Concat Fusion Common            3D Common accuracy        0.9864          0.9679                0.0185                        1.9114
action Concat Fusion Common            3D Common macro_f1        0.9751          0.9470                0.0281                        2.9673
action         Gated Fusion Concat Fusion Common accuracy        0.9781          0.9864               -0.0083                       -0.8414
action         Gated Fusion Concat Fusion Common macro_f1        0.9618          0.9751               -0.0133                       -1.3640
action         Gated Fusion            2D Common accuracy        0.9781          0.9698                0.0083                        0.8558
action         Gated Fusion            2D Common macro_f1        0.9618          0.9436                0.0182                        1.9288
action         Gated Fusion            3D Common accuracy        0.9781          0.9679                0.0102                        1.0538
action         Gated Fusion            3D Common macro_f1        0.9618          0.9470                0.0148                        1.5628

## Best Model Summary

### Binary
- **best_accuracy**: Concat Fusion Common (2D+3D, 99D) = 0.9393
- **best_macro_f1**: Concat Fusion Common (2D+3D, 99D) = 0.9299
- **best_fall_recall**: Concat Fusion Common (2D+3D, 99D) = 0.932

### Action
- **best_accuracy**: Concat Fusion Common (2D+3D, 99D) = 0.9864
- **best_macro_f1**: Concat Fusion Common (2D+3D, 99D) = 0.9751

## Interpretation Template

If Gated Fusion achieves higher Macro F1 than Concat Fusion, it suggests that adaptive weighting between 2D and estimated 3D skeleton streams is more effective than fixed early concatenation.

If Concat Fusion or Gated Fusion outperforms both 2D Common and 3D Common, it suggests that 2D and estimated 3D pose representations provide complementary information.

If 3D Common does not outperform 2D Common, this can be explained by the fact that the 3D pose is estimated from 2D keypoints rather than obtained from ground-truth 3D annotations. Errors from 2D detection and 2D-to-3D lifting may affect the quality of the 3D representation.
