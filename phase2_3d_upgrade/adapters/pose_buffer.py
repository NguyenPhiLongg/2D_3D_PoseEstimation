from collections import deque
import numpy as np


class PoseSequenceBuffer:
    """
    Store a fixed-length sequence of 2D poses for PoseFormerV2 inference.
    """

    def __init__(self, sequence_length=27):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def reset(self):
        self.buffer.clear()

    def append(self, pose_2d):
        """
        pose_2d shape: (17, 2)
        """
        pose_2d = np.asarray(pose_2d, dtype=np.float32)

        if pose_2d.shape != (17, 2):
            raise ValueError(f"Expected pose shape (17, 2), got {pose_2d.shape}")

        self.buffer.append(pose_2d)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def get_sequence(self):
        """
        Return sequence with shape (T, 17, 2)
        """
        if not self.is_ready():
            return None

        return np.asarray(self.buffer, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)