import os
import sys
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2_ROOT = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade")

if PHASE2_ROOT not in sys.path:
    sys.path.insert(0, PHASE2_ROOT)

from adapters.coco_to_h36m import coco17_to_h36m17, is_valid_coco_pose
from adapters.normalize_2d_pose import normalize_2d_pose
from adapters.pose_buffer import PoseSequenceBuffer
from inference.poseformerv2_loader import load_poseformerv2_checkpoint


class PoseFormerV2Inferencer:
    """
    End-to-end 2D-to-3D inference helper for Phase 2.

    Pipeline:
        YOLO COCO 2D keypoints
        -> approximate H36M 17-joint format
        -> normalize 2D pose
        -> buffer 27 frames
        -> PoseFormerV2 pretrained model
        -> 3D skeleton
    """

    def __init__(
        self,
        checkpoint_path=None,
        sequence_length=27,
        frame_kept=1,
        coeff_kept=3,
        device=None,
    ):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PHASE2_ROOT,
                "checkpoints",
                "1_3_27_48.7.bin"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.sequence_length = sequence_length

        self.model = load_poseformerv2_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            sequence_length=sequence_length,
            frame_kept=frame_kept,
            coeff_kept=coeff_kept,
        )

        self.buffer = PoseSequenceBuffer(sequence_length=sequence_length)

    def reset(self):
        self.buffer.reset()

    def add_coco_keypoints(self, coco_keypoints):
        """
        Add one frame of YOLO COCO keypoints.

        Input:
            coco_keypoints: numpy array, shape (17, 2)

        Output:
            ready: bool
        """

        coco_keypoints = np.asarray(coco_keypoints, dtype=np.float32)

        if coco_keypoints.shape != (17, 2):
            raise ValueError(f"Expected COCO keypoints shape (17, 2), got {coco_keypoints.shape}")

        if not is_valid_coco_pose(coco_keypoints, min_visible_points=8):
            self.reset()
            return False

        h36m_pose = coco17_to_h36m17(coco_keypoints)
        normalized_pose = normalize_2d_pose(h36m_pose)

        self.buffer.append(normalized_pose)

        return self.buffer.is_ready()

    def predict_from_buffer(self):
        """
        Predict 3D pose from the current 27-frame buffer.

        Output:
            pose_3d: numpy array with shape (17, 3)
        """

        sequence = self.buffer.get_sequence()

        if sequence is None:
            return None

        # Shape: (T, 17, 2) -> (1, T, 17, 2)
        input_tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        # Expected output shape: (1, 1, 17, 3)
        output_np = output.detach().cpu().numpy()

        if output_np.ndim == 4:
            pose_3d = output_np[0, 0]
        elif output_np.ndim == 3:
            pose_3d = output_np[0]
        else:
            raise RuntimeError(f"Unexpected PoseFormerV2 output shape: {output_np.shape}")

        return pose_3d.astype(np.float32)

    def add_and_predict(self, coco_keypoints):
        """
        Add one frame of COCO keypoints and return 3D pose if buffer is ready.

        Output:
            pose_3d or None
        """

        ready = self.add_coco_keypoints(coco_keypoints)

        if not ready:
            return None

        return self.predict_from_buffer()


def test_inferencer_with_random_input():
    """
    Quick test with random fake COCO keypoints.
    """

    inferencer = PoseFormerV2Inferencer()

    pose_3d = None

    for _ in range(27):
        fake_coco = np.random.rand(17, 2).astype(np.float32) * 500
        pose_3d = inferencer.add_and_predict(fake_coco)

    if pose_3d is None:
        raise RuntimeError("Pose 3D was not generated.")

    print("3D pose generated successfully.")
    print("Pose 3D shape:", pose_3d.shape)
    print("Pose 3D min/max:", pose_3d.min(), pose_3d.max())


if __name__ == "__main__":
    test_inferencer_with_random_input()