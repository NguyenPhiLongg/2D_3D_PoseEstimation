import numpy as np


# COCO 17 keypoints from YOLOv8-Pose:
# 0 nose
# 1 left_eye
# 2 right_eye
# 3 left_ear
# 4 right_ear
# 5 left_shoulder
# 6 right_shoulder
# 7 left_elbow
# 8 right_elbow
# 9 left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle


def midpoint(point_a, point_b):
    return (point_a + point_b) / 2.0


def coco17_to_h36m17(coco_keypoints):
    """
    Convert YOLOv8-Pose COCO 17 keypoints to an approximate H36M-style 17-joint format.

    Input:
        coco_keypoints: numpy array with shape (17, 2)

    Output:
        h36m_keypoints: numpy array with shape (17, 2)

    Note:
        This is an approximate conversion for demo/inference visualization.
        The current project does not use Human3.6M ground truth.
    """

    coco_keypoints = np.asarray(coco_keypoints, dtype=np.float32)

    if coco_keypoints.shape != (17, 2):
        raise ValueError(f"Expected COCO keypoints shape (17, 2), got {coco_keypoints.shape}")

    h36m = np.zeros((17, 2), dtype=np.float32)

    left_shoulder = coco_keypoints[5]
    right_shoulder = coco_keypoints[6]
    left_hip = coco_keypoints[11]
    right_hip = coco_keypoints[12]

    pelvis = midpoint(left_hip, right_hip)
    thorax = midpoint(left_shoulder, right_shoulder)
    spine = midpoint(pelvis, thorax)
    neck = thorax
    head = coco_keypoints[0]

    # H36M-style joint order:
    # 0 pelvis
    # 1 right_hip
    # 2 right_knee
    # 3 right_ankle
    # 4 left_hip
    # 5 left_knee
    # 6 left_ankle
    # 7 spine
    # 8 thorax
    # 9 neck
    # 10 head
    # 11 left_shoulder
    # 12 left_elbow
    # 13 left_wrist
    # 14 right_shoulder
    # 15 right_elbow
    # 16 right_wrist

    h36m[0] = pelvis

    h36m[1] = coco_keypoints[12]
    h36m[2] = coco_keypoints[14]
    h36m[3] = coco_keypoints[16]

    h36m[4] = coco_keypoints[11]
    h36m[5] = coco_keypoints[13]
    h36m[6] = coco_keypoints[15]

    h36m[7] = spine
    h36m[8] = thorax
    h36m[9] = neck
    h36m[10] = head

    h36m[11] = coco_keypoints[5]
    h36m[12] = coco_keypoints[7]
    h36m[13] = coco_keypoints[9]

    h36m[14] = coco_keypoints[6]
    h36m[15] = coco_keypoints[8]
    h36m[16] = coco_keypoints[10]

    return h36m


def is_valid_coco_pose(coco_keypoints, min_visible_points=8):
    """
    Check whether the detected 2D pose has enough valid keypoints.

    Missing keypoints from YOLO are usually represented as (0, 0),
    not NaN in this project.
    """

    coco_keypoints = np.asarray(coco_keypoints, dtype=np.float32)

    valid_points = 0

    for x, y in coco_keypoints:
        if x > 1 and y > 1:
            valid_points += 1

    return valid_points >= min_visible_points