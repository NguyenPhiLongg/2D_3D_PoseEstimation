import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog
from collections import deque

import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Missing dependency: pyyaml. Install it with: pip install pyyaml"
    ) from exc


# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE1_DIR = os.path.join(PROJECT_ROOT, "phase1_2d_baseline")
sys.path.insert(0, PHASE1_DIR)

from model_2d import FallCNNLSTM


# =========================
# CONFIG LOADER
# =========================

DEFAULT_CONFIG = {
    "window": {
        "title": "Hierarchical Fall Detection Demo",
        "display_width": 960,
        "display_height": 540,
    },
    "camera": {
        "webcam_id": 0,
    },
    "models": {
        "yolo_model_path": "yolov8m-pose.pt",
        "binary_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_binary.pt",
        "action_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_action.pt",
        "use_cuda": True,
    },
    "prediction": {
        "sequence_length": 60,
        "fall_threshold": 0.85,
        "fall_confirm_count": 10,
        "action_names": ["Sitting", "Sleeping", "Standing", "Walking"],
    },
    "ui": {
        "show_fall_warning_on_video": True,
        "draw_pose": True,
        "draw_bbox": True,
    },
}


def deep_update(base, updates):
    """Recursively update a dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "demo_interface", "config.yaml")

    config = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in DEFAULT_CONFIG.items()
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = deep_update(config, user_config)
        print("Loaded config:", config_path)
    else:
        print("WARNING: config.yaml not found. Using default config.")

    return config


def resolve_project_path(path_value):
    """Convert a relative project path to an absolute path."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


CONFIG = load_config()

WINDOW_TITLE = CONFIG["window"]["title"]
DISPLAY_WIDTH = int(CONFIG["window"]["display_width"])
DISPLAY_HEIGHT = int(CONFIG["window"]["display_height"])

WEBCAM_ID = int(CONFIG["camera"]["webcam_id"])

YOLO_MODEL_PATH = resolve_project_path(CONFIG["models"]["yolo_model_path"])
BINARY_CHECKPOINT_PATH = resolve_project_path(CONFIG["models"]["binary_checkpoint_path"])
ACTION_CHECKPOINT_PATH = resolve_project_path(CONFIG["models"]["action_checkpoint_path"])
USE_CUDA = bool(CONFIG["models"]["use_cuda"])

SEQUENCE_LENGTH = int(CONFIG["prediction"]["sequence_length"])
FALL_THRESHOLD = float(CONFIG["prediction"]["fall_threshold"])
FALL_CONFIRM_COUNT = int(CONFIG["prediction"]["fall_confirm_count"])
ACTION_NAMES = CONFIG["prediction"]["action_names"]

SHOW_FALL_WARNING_ON_VIDEO = bool(CONFIG["ui"]["show_fall_warning_on_video"])
DRAW_POSE = bool(CONFIG["ui"]["draw_pose"])
DRAW_BBOX = bool(CONFIG["ui"]["draw_bbox"])

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features_from_keypoints(keypoints, frame_width, frame_height, prev_center=None):
    keypoints = keypoints.astype(np.float32)

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)

    width = max_x - min_x
    height = max_y - min_y

    eps = 1e-6

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    scale = max(width, height) + eps

    norm_xs = (xs - center_x) / scale
    norm_ys = (ys - center_y) / scale

    normalized_coords = np.empty((17, 2), dtype=np.float32)
    normalized_coords[:, 0] = norm_xs
    normalized_coords[:, 1] = norm_ys
    normalized_coords = normalized_coords.flatten()

    aspect_ratio = height / (width + eps)
    norm_width = width / scale
    norm_height = height / scale

    center_x_norm = center_x / frame_width
    center_y_norm = center_y / frame_height

    if prev_center is None:
        velocity = 0.0
    else:
        prev_x, prev_y = prev_center
        velocity = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

    handcrafted = np.array(
        [
            aspect_ratio,
            norm_width,
            norm_height,
            center_x_norm,
            center_y_norm,
            velocity
        ],
        dtype=np.float32
    )

    features = np.concatenate([normalized_coords, handcrafted], axis=0)

    current_center = (center_x, center_y)
    bbox = (min_x, min_y, max_x, max_y)

    return features, current_center, bbox, aspect_ratio


# =========================
# DRAWING
# =========================

def resize_with_padding(frame, target_width=960, target_height=540):
    h, w = frame.shape[:2]

    scale = min(target_width / w, target_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def draw_pose(frame, keypoints):
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    for p1, p2 in SKELETON:
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 0),
                2
            )


def draw_warning_on_frame(frame, final_label):
    if final_label == "Fall" and SHOW_FALL_WARNING_ON_VIDEO:
        h, _ = frame.shape[:2]

        cv2.rectangle(frame, (10, h - 65), (510, h - 15), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "WARNING: FALL DETECTED!",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


# =========================
# LOAD MODEL
# =========================

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = checkpoint.get("input_dim", 40)
    num_classes = checkpoint.get("num_classes", 2)
    class_names = checkpoint.get("class_names", [])

    model = FallCNNLSTM(
        input_dim=input_dim,
        num_classes=num_classes,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded model:", checkpoint_path)
    print("Input dim:", input_dim)
    print("Num classes:", num_classes)
    print("Class names:", class_names)

    return model


# =========================
# APP
# =========================

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)

        if USE_CUDA and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("Device:", self.device)

        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        self.binary_model = load_model(BINARY_CHECKPOINT_PATH, self.device)
        self.action_model = load_model(ACTION_CHECKPOINT_PATH, self.device)

        self.cap = None
        self.source_type = "webcam"
        self.source_name = "Webcam"
        self.video_ended = False

        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prev_center = None
        self.fall_counter = 0

        self.final_label = "WAITING"
        self.fall_prob = 0.0
        self.action_prob = 0.0
        self.aspect_ratio = 0.0

        self.is_running = True

        self.video_label = tk.Label(
            self.root,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg="black"
        )
        self.video_label.pack(padx=10, pady=10)

        self.info_label = tk.Label(
            self.root,
            text=f"Prediction: WAITING | Fall prob: 0.00 | Action prob: 0.00 | Buffer: 0/{SEQUENCE_LENGTH}",
            font=("Arial", 14, "bold"),
            fg="blue"
        )
        self.info_label.pack(pady=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.upload_button = tk.Button(
            button_frame,
            text="Upload Video",
            command=self.upload_video,
            width=18,
            height=2
        )
        self.upload_button.grid(row=0, column=0, padx=10)

        self.webcam_button = tk.Button(
            button_frame,
            text="Return to Webcam",
            command=self.return_to_webcam,
            width=18,
            height=2
        )
        self.webcam_button.grid(row=0, column=1, padx=10)

        self.quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self.close_app,
            width=18,
            height=2
        )
        self.quit_button.grid(row=0, column=2, padx=10)

        self.status_label = tk.Label(
            self.root,
            text="Status: Webcam running",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=5)

        self.open_webcam()
        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def reset_sequence_state(self):
        self.sequence_buffer.clear()
        self.prev_center = None
        self.final_label = "WAITING"
        self.fall_prob = 0.0
        self.action_prob = 0.0
        self.aspect_ratio = 0.0
        self.fall_counter = 0

    def release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_webcam(self):
        self.release_capture()
        self.reset_sequence_state()

        self.cap = cv2.VideoCapture(WEBCAM_ID)
        self.source_type = "webcam"
        self.source_name = "Webcam"
        self.video_ended = False

        self.status_label.config(text="Status: Webcam running")

        if not self.cap.isOpened():
            self.status_label.config(text="ERROR: Cannot open webcam")
            print("ERROR: Cannot open webcam")

    def upload_video(self):
        video_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not video_path:
            return

        self.release_capture()
        self.reset_sequence_state()

        self.cap = cv2.VideoCapture(video_path)
        self.source_type = "video"
        self.source_name = os.path.basename(video_path)
        self.video_ended = False

        if not self.cap.isOpened():
            self.status_label.config(text="ERROR: Cannot open selected video")
            print("ERROR: Cannot open selected video:", video_path)
            self.open_webcam()
            return

        self.status_label.config(text=f"Status: Uploaded video running - {self.source_name}")
        print("Uploaded video:", video_path)

    def return_to_webcam(self):
        print("Returning to webcam...")
        self.open_webcam()

    def predict_hierarchical(self, sequence_tensor):
        with torch.no_grad():
            binary_logits = self.binary_model(sequence_tensor)
            binary_probs = torch.softmax(binary_logits, dim=1)[0]

            fall_prob = binary_probs[1].item()

            action_logits = self.action_model(sequence_tensor)
            action_probs = torch.softmax(action_logits, dim=1)[0]

            action_class = torch.argmax(action_probs).item()
            action_prob = action_probs[action_class].item()
            action_name = ACTION_NAMES[action_class]

            if fall_prob >= FALL_THRESHOLD:
                self.fall_counter += 1
            else:
                self.fall_counter = 0

            if self.fall_counter >= FALL_CONFIRM_COUNT:
                return "Fall", fall_prob, action_prob

            return action_name, fall_prob, action_prob

    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]

        results = self.yolo_model(frame, verbose=False, device=self.device)

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].detach().cpu().numpy()

            features, self.prev_center, bbox, self.aspect_ratio = extract_features_from_keypoints(
                keypoints,
                frame_width,
                frame_height,
                self.prev_center
            )

            self.sequence_buffer.append(features)

            if DRAW_POSE:
                draw_pose(frame, keypoints)

            min_x, min_y, max_x, max_y = bbox

            if DRAW_BBOX:
                cv2.rectangle(
                    frame,
                    (int(min_x), int(min_y)),
                    (int(max_x), int(max_y)),
                    (255, 0, 0),
                    2
                )

            if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                sequence_np = np.array(self.sequence_buffer, dtype=np.float32)
                sequence_tensor = torch.tensor(sequence_np).unsqueeze(0).to(self.device)

                self.final_label, self.fall_prob, self.action_prob = self.predict_hierarchical(
                    sequence_tensor
                )

        else:
            self.final_label = "NO PERSON"
            self.fall_prob = 0.0
            self.action_prob = 0.0
            self.prev_center = None
            self.fall_counter = 0

        return frame

    def update_info_label(self):
        if self.final_label == "Fall":
            text_color = "red"
        elif self.final_label in ACTION_NAMES:
            text_color = "green"
        elif self.final_label == "NO PERSON":
            text_color = "orange"
        else:
            text_color = "blue"

        self.info_label.config(
            text=(
                f"Prediction: {self.final_label} | "
                f"Fall prob: {self.fall_prob:.2f} | "
                f"Action prob: {self.action_prob:.2f} | "
                f"Buffer: {len(self.sequence_buffer)}/{SEQUENCE_LENGTH} | "
                f"Source: {self.source_name}"
            ),
            fg=text_color
        )

    def update_frame(self):
        if not self.is_running:
            return

        if self.video_ended:
            self.update_info_label()
            self.root.after(300, self.update_frame)
            return

        if self.cap is None or not self.cap.isOpened():
            self.root.after(30, self.update_frame)
            return

        success, frame = self.cap.read()

        if not success:
            if self.source_type == "video":
                self.video_ended = True
                self.status_label.config(
                    text="Status: Video ended. Press 'Return to Webcam' to switch back."
                )
                print("Video ended. Waiting for user to return to webcam.")
                self.release_capture()
            else:
                self.status_label.config(text="ERROR: Cannot read webcam frame")

            self.root.after(300, self.update_frame)
            return

        frame = self.process_frame(frame)

        frame = resize_with_padding(
            frame,
            target_width=DISPLAY_WIDTH,
            target_height=DISPLAY_HEIGHT
        )

        draw_warning_on_frame(frame, self.final_label)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image=image)

        self.video_label.imgtk = image_tk
        self.video_label.configure(image=image_tk)

        self.update_info_label()

        self.root.after(1, self.update_frame)

    def close_app(self):
        self.is_running = False
        self.release_capture()
        self.root.destroy()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()