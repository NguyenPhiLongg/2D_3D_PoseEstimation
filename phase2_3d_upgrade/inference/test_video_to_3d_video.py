import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2_ROOT = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade")

if PHASE2_ROOT not in sys.path:
    sys.path.insert(0, PHASE2_ROOT)

from inference.infer_3d_pose import PoseFormerV2Inferencer
from inference.visualize_3d import render_3d_pose_to_image_array


YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8m-pose.pt")
INPUT_VIDEO_PATH = r"C:\Users\ASUS\Downloads\3D_Human_Pose_NCKH\data\1_raw_videos\Fall\f_raw_s_1\S_D_0166.mp4"
# INPUT_VIDEO_PATH = os.path.join(
#     PROJECT_ROOT,
#     "data",
#     "1_raw_videos",
#     "Not_Fall",
#     "Walking",
#     "Walking While Using Phone (82).mp4"
# )
#
# OUTPUT_VIDEO_PATH = os.path.join(
#     PHASE2_ROOT,
#     "outputs",
#     "demo_3d_videos",
#     "walking_82_2d_3d_demo.mp4"
# )
OUTPUT_VIDEO_PATH = os.path.join(
    PHASE2_ROOT,
    "outputs",
    "demo_3d_videos",
    "fall_s_d_0166_2d_3d_demo.mp4"
)

def draw_2d_pose(frame, keypoints):
    skeleton = [
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    for x, y in keypoints:
        if x > 1 and y > 1:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    for a, b in skeleton:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        if xa > 1 and ya > 1 and xb > 1 and yb > 1:
            cv2.line(
                frame,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                (255, 255, 0),
                2
            )

    return frame


def resize_with_padding(image, target_width, target_height):
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, scale, x_offset, y_offset


def create_3d_video_from_real_video(
    video_path=INPUT_VIDEO_PATH,
    output_path=OUTPUT_VIDEO_PATH,
    max_frames=300
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Input video:", video_path)
    print("Output video:", output_path)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    inferencer = PoseFormerV2Inferencer()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    left_width = 800
    right_width = 480
    output_height = 480
    output_width = left_width + right_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )

    frame_idx = 0
    valid_pose_count = 0
    generated_3d_count = 0

    last_3d_image = np.zeros((output_height, right_width, 3), dtype=np.uint8)

    while cap.isOpened() and frame_idx < max_frames:
        success, frame = cap.read()

        if not success:
            break

        frame_idx += 1

        display_2d, scale, x_offset, y_offset = resize_with_padding(
            frame,
            left_width,
            output_height
        )

        results = yolo_model(frame, verbose=False)

        pose_3d = None

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy

            if keypoints_xy is not None and len(keypoints_xy) > 0:
                coco_keypoints = keypoints_xy[0].detach().cpu().numpy()
                valid_pose_count += 1

                display_keypoints = coco_keypoints.copy()
                display_keypoints[:, 0] = display_keypoints[:, 0] * scale + x_offset
                display_keypoints[:, 1] = display_keypoints[:, 1] * scale + y_offset

                display_2d = draw_2d_pose(display_2d, display_keypoints)

                pose_3d = inferencer.add_and_predict(coco_keypoints)

        if pose_3d is not None:
            generated_3d_count += 1

            last_3d_image = render_3d_pose_to_image_array(
                pose_3d,
                width=right_width,
                height=output_height
            )

        combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined[:, :left_width] = display_2d
        combined[:, left_width:] = last_3d_image

        cv2.putText(
            combined,
            "2D YOLO Pose",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            combined,
            "PoseFormerV2 3D Pose",
            (left_width + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            combined,
            f"Frame: {frame_idx} | Valid 2D: {valid_pose_count} | 3D generated: {generated_3d_count}",
            (20, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if generated_3d_count == 0:
            cv2.putText(
                combined,
                "Waiting for 27 valid frames...",
                (left_width + 35, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        writer.write(combined)

        if frame_idx % 30 == 0:
            print(
                f"Processed: {frame_idx} | "
                f"Valid 2D: {valid_pose_count} | "
                f"Generated 3D: {generated_3d_count}"
            )

    cap.release()
    writer.release()

    print("Done.")
    print("Processed frames:", frame_idx)
    print("Valid 2D pose frames:", valid_pose_count)
    print("Generated 3D frames:", generated_3d_count)
    print("Saved 2D + 3D demo video to:", output_path)


if __name__ == "__main__":
    create_3d_video_from_real_video()