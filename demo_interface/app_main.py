import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog
from collections import deque
import copy

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
PHASE2_DIR = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade")
PHASE2_INFERENCE_DIR = os.path.join(PHASE2_DIR, "inference")

sys.path.insert(0, PHASE1_DIR)
sys.path.insert(0, PHASE2_DIR)
sys.path.insert(0, PHASE2_INFERENCE_DIR)

from model_2d import FallCNNLSTM
from model_3d import FallCNNLSTM3D

try:
    from inference.infer_3d_pose import PoseFormerV2Inferencer
except Exception as exc:
    PoseFormerV2Inferencer = None
    POSEFORMER_IMPORT_ERROR = exc


# =========================
# DEFAULT CONFIG
# =========================

DEFAULT_CONFIG = {
    "window": {
        "title": "2D / 3D / Fusion Fall Detection Demo",
        "display_width": 960,
        "display_height": 540,
    },
    "camera": {
        "webcam_id": 0,
    },
    "models": {
        "yolo_model_path": "yolov8m-pose.pt",

        # 2D checkpoints
        "binary_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_binary.pt",
        "action_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_action.pt",
        "two_d_binary_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_binary.pt",
        "two_d_action_checkpoint_path": "phase1_2d_baseline/checkpoints/best_model_2d_action.pt",

        # 3D checkpoints
        "three_d_binary_checkpoint_path": "phase2_3d_upgrade/checkpoints/best_model_3d_binary.pt",
        "three_d_action_checkpoint_path": "phase2_3d_upgrade/checkpoints/best_model_3d_action.pt",

        # Fusion checkpoints
        "fusion_binary_checkpoint_path": "phase2_3d_upgrade/checkpoints/best_model_fusion_2d3d_binary.pt",
        "fusion_action_checkpoint_path": "phase2_3d_upgrade/checkpoints/best_model_fusion_2d3d_action.pt",

        # PoseFormerV2 checkpoint
        "poseformer_checkpoint_path": "phase2_3d_upgrade/checkpoints/1_3_27_48.7.bin",

        "use_cuda": True,
    },
    "prediction": {
        # Options:
        #   "2d"     : YOLO 2D keypoints -> 2D model
        #   "3d"     : YOLO 2D keypoints -> PoseFormerV2 -> 3D model
        #   "fusion" : 2D features + 3D features -> Fusion model
        "demo_mode": "fusion",

        "sequence_length": 60,
        "fall_threshold": 0.85,
        "fall_confirm_count": 10,
        "action_names": ["Sitting", "Sleeping", "Standing", "Walking"],
    },
    "features": {
        "reference_width": 1920.0,
        "reference_height": 1080.0,
        "depth_scale": 1.0,
    },
    "ui": {
        "show_fall_warning_on_video": True,
        "draw_pose": True,
        "draw_bbox": True,
    },
}


# =========================
# CONFIG LOADER
# =========================

def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value

    return base


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "demo_interface", "config.yaml")

    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        config = deep_update(config, user_config)
        print("Loaded config:", config_path)
    else:
        print("WARNING: config.yaml not found. Using default config.")

    return config


def resolve_project_path(path_value):
    if os.path.isabs(path_value):
        return path_value

    return os.path.join(PROJECT_ROOT, path_value)


CONFIG = load_config()

WINDOW_TITLE = CONFIG["window"]["title"]
DISPLAY_WIDTH = int(CONFIG["window"]["display_width"])
DISPLAY_HEIGHT = int(CONFIG["window"]["display_height"])

WEBCAM_ID = int(CONFIG["camera"]["webcam_id"])

YOLO_MODEL_PATH = resolve_project_path(CONFIG["models"]["yolo_model_path"])
POSEFORMER_CHECKPOINT_PATH = resolve_project_path(CONFIG["models"]["poseformer_checkpoint_path"])

USE_CUDA = bool(CONFIG["models"]["use_cuda"])

DEMO_MODE = str(CONFIG["prediction"].get("demo_mode", "fusion")).lower().strip()

SEQUENCE_LENGTH = int(CONFIG["prediction"]["sequence_length"])
FALL_THRESHOLD = float(CONFIG["prediction"]["fall_threshold"])
FALL_CONFIRM_COUNT = int(CONFIG["prediction"]["fall_confirm_count"])
ACTION_NAMES = CONFIG["prediction"]["action_names"]

REFERENCE_WIDTH = float(CONFIG["features"].get("reference_width", 1920.0))
REFERENCE_HEIGHT = float(CONFIG["features"].get("reference_height", 1080.0))
DEPTH_SCALE = float(CONFIG["features"].get("depth_scale", 1.0))

SHOW_FALL_WARNING_ON_VIDEO = bool(CONFIG["ui"]["show_fall_warning_on_video"])
DRAW_POSE = bool(CONFIG["ui"]["draw_pose"])
DRAW_BBOX = bool(CONFIG["ui"]["draw_bbox"])

VALID_DEMO_MODES = ["2d", "3d", "fusion"]

if DEMO_MODE not in VALID_DEMO_MODES:
    raise ValueError(
        f"Invalid demo_mode: {DEMO_MODE}. "
        f"Valid values are: {VALID_DEMO_MODES}"
    )


def get_checkpoint_paths_for_mode(mode):
    if mode == "2d":
        binary_path = CONFIG["models"].get(
            "two_d_binary_checkpoint_path",
            CONFIG["models"].get("binary_checkpoint_path")
        )
        action_path = CONFIG["models"].get(
            "two_d_action_checkpoint_path",
            CONFIG["models"].get("action_checkpoint_path")
        )

    elif mode == "3d":
        binary_path = CONFIG["models"]["three_d_binary_checkpoint_path"]
        action_path = CONFIG["models"]["three_d_action_checkpoint_path"]

    elif mode == "fusion":
        binary_path = CONFIG["models"]["fusion_binary_checkpoint_path"]
        action_path = CONFIG["models"]["fusion_action_checkpoint_path"]

    else:
        raise ValueError(f"Unknown demo mode: {mode}")

    return resolve_project_path(binary_path), resolve_project_path(action_path)


BINARY_CHECKPOINT_PATH, ACTION_CHECKPOINT_PATH = get_checkpoint_paths_for_mode(DEMO_MODE)


# =========================
# SKELETON CONNECTIONS
# =========================

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
# SAFE TORCH LOAD
# =========================

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# FEATURE EXTRACTION - 2D
# =========================

def extract_features_2d_from_keypoints(keypoints, prev_center=None):
    """
    2D feature extraction.

    This follows the new training pipeline:
        per-frame x/y mean-std normalization.

    Output:
        34 normalized 2D keypoint features
        + 6 handcrafted 2D features
        = 40 features
    """

    keypoints = np.asarray(keypoints, dtype=np.float32)

    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    width = max_x - min_x
    height = max_y - min_y

    eps = 1e-6

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    pose_2d = keypoints.reshape(17, 2)

    mean_2d = pose_2d.mean(axis=0, keepdims=True)
    std_2d = pose_2d.std(axis=0, keepdims=True)

    std_2d = np.where(std_2d < eps, 1.0, std_2d)

    normalized_pose_2d = (pose_2d - mean_2d) / std_2d
    normalized_coords = normalized_pose_2d.reshape(-1).astype(np.float32)

    aspect_ratio = height / (width + eps)

    scale = max(width, height) + eps
    norm_width = width / (scale + eps)
    norm_height = height / (scale + eps)

    center_x_norm = center_x / REFERENCE_WIDTH
    center_y_norm = center_y / REFERENCE_HEIGHT

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

    features = np.concatenate([normalized_coords, handcrafted], axis=0).astype(np.float32)

    if features.shape[0] != 40:
        raise ValueError(f"2D feature dim must be 40, got {features.shape[0]}")

    current_center = (center_x, center_y)
    bbox = (min_x, min_y, max_x, max_y)

    return features, current_center, bbox, aspect_ratio


# =========================
# FEATURE EXTRACTION - 3D
# =========================

def normalize_pose3d_mean_std(pose3d):
    """
    Normalize one 3D pose frame using per-axis mean/std.

    Formula:
        pose = (pose - pose.mean(axis=0)) / pose.std(axis=0)

    Mean/std are computed separately for x, y, z.
    """

    pose3d = np.asarray(pose3d, dtype=np.float32)

    if pose3d.shape != (17, 3):
        raise ValueError(f"Expected pose3d shape (17, 3), got {pose3d.shape}")

    eps = 1e-6

    mean = pose3d.mean(axis=0, keepdims=True)
    std = pose3d.std(axis=0, keepdims=True)

    std = np.where(std < eps, 1.0, std)

    normalized = (pose3d - mean) / std

    return normalized.astype(np.float32)


def extract_features_3d_from_pose(pose3d, prev_center=None):
    """
    3D feature extraction.

    Output:
        51 normalized 3D keypoint features
        + 8 handcrafted 3D features
        = 59 features
    """

    pose = normalize_pose3d_mean_std(pose3d)

    xs = pose[:, 0]
    ys = pose[:, 1]
    zs = pose[:, 2]

    min_x = np.min(xs)
    max_x = np.max(xs)

    min_y = np.min(ys)
    max_y = np.max(ys)

    min_z = np.min(zs)
    max_z = np.max(zs)

    width_x = max_x - min_x
    depth_y = max_y - min_y
    height_z = max_z - min_z

    eps = 1e-6

    height_width_ratio = height_z / (width_x + eps)
    depth_width_ratio = depth_y / (width_x + eps)

    # Joint 10 = head, joint 0 = pelvis in H36M-like layout
    head_height = zs[10] - zs[0]

    # Joint 8 = thorax, joint 0 = pelvis
    torso_vector = pose[8, :] - pose[0, :]
    torso_horizontal = np.linalg.norm(torso_vector[0:2])
    torso_vertical = abs(torso_vector[2]) + eps
    torso_tilt = torso_horizontal / torso_vertical

    center = pose.mean(axis=0)

    if prev_center is None:
        velocity = 0.0
    else:
        velocity = float(np.linalg.norm(center - prev_center))

    handcrafted = np.array(
        [
            width_x,
            depth_y,
            height_z,
            height_width_ratio,
            depth_width_ratio,
            head_height,
            torso_tilt,
            velocity
        ],
        dtype=np.float32
    )

    coords = pose.reshape(-1).astype(np.float32)
    features = np.concatenate([coords, handcrafted], axis=0).astype(np.float32)

    if features.shape[0] != 59:
        raise ValueError(f"3D feature dim must be 59, got {features.shape[0]}")

    return features, center


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

        cv2.rectangle(frame, (10, h - 65), (530, h - 15), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "WARNING: FALL DETECTED!",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


def draw_mode_on_frame(frame, mode):
    text = f"Mode: {mode.upper()}"

    cv2.rectangle(frame, (10, 10), (250, 45), (0, 0, 0), -1)

    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# =========================
# MODEL LOADING
# =========================

def load_sequence_model(checkpoint_path, device, model_class):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, device)

    input_dim = checkpoint.get("input_dim", 40)
    num_classes = checkpoint.get("num_classes", 2)
    class_names = checkpoint.get("class_names", [])

    model = model_class(
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
    print("Model class:", model_class.__name__)
    print("Input dim:", input_dim)
    print("Num classes:", num_classes)
    print("Class names:", class_names)

    return model, input_dim, class_names


def create_poseformer_inferencer(checkpoint_path, device):
    if PoseFormerV2Inferencer is None:
        raise ImportError(
            "Cannot import PoseFormerV2Inferencer. "
            f"Original error: {POSEFORMER_IMPORT_ERROR}"
        )

    device_str = "cuda" if device.type == "cuda" else "cpu"

    try:
        return PoseFormerV2Inferencer(
            checkpoint_path=str(checkpoint_path),
            device=device_str
        )
    except TypeError:
        pass

    try:
        return PoseFormerV2Inferencer(
            ckpt_path=str(checkpoint_path),
            device=device_str
        )
    except TypeError:
        pass

    try:
        return PoseFormerV2Inferencer(
            model_path=str(checkpoint_path),
            device=device_str
        )
    except TypeError:
        pass

    try:
        return PoseFormerV2Inferencer(str(checkpoint_path), device_str)
    except TypeError:
        pass

    try:
        return PoseFormerV2Inferencer(str(checkpoint_path))
    except TypeError:
        pass

    return PoseFormerV2Inferencer()


# =========================
# APP
# =========================

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)

        if USE_CUDA and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.yolo_device = 0
        else:
            self.device = torch.device("cpu")
            self.yolo_device = "cpu"

        print("Device:", self.device)
        print("YOLO device:", self.yolo_device)
        print("Demo mode:", DEMO_MODE)

        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        if DEMO_MODE == "2d":
            model_class = FallCNNLSTM
        elif DEMO_MODE in ["3d", "fusion"]:
            model_class = FallCNNLSTM3D
        else:
            raise ValueError(f"Invalid demo mode: {DEMO_MODE}")

        self.binary_model, self.binary_input_dim, self.binary_class_names = load_sequence_model(
            BINARY_CHECKPOINT_PATH,
            self.device,
            model_class
        )

        self.action_model, self.action_input_dim, self.action_class_names = load_sequence_model(
            ACTION_CHECKPOINT_PATH,
            self.device,
            model_class
        )

        if self.binary_input_dim != self.action_input_dim:
            print("WARNING: Binary and action model input dimensions are different.")
            print("Binary input dim:", self.binary_input_dim)
            print("Action input dim:", self.action_input_dim)

        self.expected_input_dim = self.binary_input_dim

        self.poseformer = None

        if DEMO_MODE in ["3d", "fusion"]:
            print("Loading PoseFormerV2 checkpoint:", POSEFORMER_CHECKPOINT_PATH)
            self.poseformer = create_poseformer_inferencer(
                POSEFORMER_CHECKPOINT_PATH,
                self.device
            )
            print("PoseFormerV2 loaded.")

        self.cap = None
        self.source_type = "webcam"
        self.source_name = "Webcam"
        self.video_ended = False

        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

        self.prev_center_2d = None
        self.prev_center_3d = None

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
            text=f"Prediction: WAITING | Mode: {DEMO_MODE.upper()} | Buffer: 0/{SEQUENCE_LENGTH}",
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
            text=f"Status: Webcam running | Mode: {DEMO_MODE.upper()}",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=5)

        self.open_webcam()
        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def reset_poseformer_state(self):
        if DEMO_MODE not in ["3d", "fusion"]:
            return

        if self.poseformer is None:
            return

        if hasattr(self.poseformer, "reset"):
            try:
                self.poseformer.reset()
                return
            except Exception:
                pass

        try:
            self.poseformer = create_poseformer_inferencer(
                POSEFORMER_CHECKPOINT_PATH,
                self.device
            )
        except Exception as exc:
            print("WARNING: Cannot reset PoseFormerV2:", exc)

    def reset_sequence_state(self):
        self.sequence_buffer.clear()

        self.prev_center_2d = None
        self.prev_center_3d = None

        self.final_label = "WAITING"
        self.fall_prob = 0.0
        self.action_prob = 0.0
        self.aspect_ratio = 0.0
        self.fall_counter = 0

        self.reset_poseformer_state()

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

        self.status_label.config(
            text=f"Status: Webcam running | Mode: {DEMO_MODE.upper()}"
        )

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

        self.status_label.config(
            text=f"Status: Uploaded video running - {self.source_name} | Mode: {DEMO_MODE.upper()}"
        )
        print("Uploaded video:", video_path)

    def return_to_webcam(self):
        print("Returning to webcam...")
        self.open_webcam()

    def build_model_features(self, keypoints):
        """
        Build features according to selected demo mode.

        2D mode:
            40 features

        3D mode:
            59 features

        Fusion mode:
            99 features = 40 2D + 59 3D
        """

        features_2d, self.prev_center_2d, bbox, self.aspect_ratio = extract_features_2d_from_keypoints(
            keypoints,
            self.prev_center_2d
        )

        if DEMO_MODE == "2d":
            return features_2d, bbox, "OK"

        if self.poseformer is None:
            return None, bbox, "NO POSEFORMER"

        pose3d = self.poseformer.add_and_predict(keypoints)

        if pose3d is None:
            return None, bbox, "BUILDING 3D"

        pose3d = np.asarray(pose3d, dtype=np.float32)

        if pose3d.shape != (17, 3):
            return None, bbox, "INVALID 3D"

        features_3d, self.prev_center_3d = extract_features_3d_from_pose(
            pose3d,
            self.prev_center_3d
        )

        if DEMO_MODE == "3d":
            return features_3d, bbox, "OK"

        if DEMO_MODE == "fusion":
            fusion_features = np.concatenate([features_2d, features_3d], axis=0).astype(np.float32)

            if fusion_features.shape[0] != 99:
                raise ValueError(f"Fusion feature dim must be 99, got {fusion_features.shape[0]}")

            return fusion_features, bbox, "OK"

        raise ValueError(f"Invalid demo mode: {DEMO_MODE}")

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
        results = self.yolo_model(frame, verbose=False, device=self.yolo_device)

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].detach().cpu().numpy().astype(np.float32)

            features, bbox, feature_status = self.build_model_features(keypoints)

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

            if feature_status != "OK":
                self.final_label = feature_status
                self.fall_prob = 0.0
                self.action_prob = 0.0
                return frame

            if features is None:
                return frame

            if features.shape[0] != self.expected_input_dim:
                raise ValueError(
                    f"Feature dimension mismatch. "
                    f"Expected {self.expected_input_dim}, got {features.shape[0]}"
                )

            self.sequence_buffer.append(features)

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
            self.prev_center_2d = None
            self.prev_center_3d = None
            self.fall_counter = 0

        return frame

    def update_info_label(self):
        if self.final_label == "Fall":
            text_color = "red"
        elif self.final_label in ACTION_NAMES:
            text_color = "green"
        elif self.final_label == "NO PERSON":
            text_color = "orange"
        elif self.final_label in ["BUILDING 3D", "INVALID 3D", "NO POSEFORMER"]:
            text_color = "purple"
        else:
            text_color = "blue"

        self.info_label.config(
            text=(
                f"Prediction: {self.final_label} | "
                f"Mode: {DEMO_MODE.upper()} | "
                f"Fall prob: {self.fall_prob:.2f} | "
                f"Action prob: {self.action_prob:.2f} | "
                f"Buffer: {len(self.sequence_buffer)}/{SEQUENCE_LENGTH} | "
                f"Input dim: {self.expected_input_dim} | "
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

        draw_mode_on_frame(frame, DEMO_MODE)
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
    print("=" * 80)
    print("2D / 3D / Fusion Fall Detection Demo")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)
    print("Demo mode:", DEMO_MODE)
    print("YOLO model:", YOLO_MODEL_PATH)
    print("Binary checkpoint:", BINARY_CHECKPOINT_PATH)
    print("Action checkpoint:", ACTION_CHECKPOINT_PATH)

    if DEMO_MODE in ["3d", "fusion"]:
        print("PoseFormer checkpoint:", POSEFORMER_CHECKPOINT_PATH)

    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()