import os
import sys
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POSEFORMERV2_REPO = os.path.join(
    PROJECT_ROOT,
    "phase2_3d_upgrade",
    "poseformerv2_repo"
)

if POSEFORMERV2_REPO not in sys.path:
    sys.path.insert(0, POSEFORMERV2_REPO)

from common.model_poseformer import PoseTransformerV2


def build_poseformerv2_model(
    sequence_length=27,
    frame_kept=1,
    coeff_kept=3,
    depth=4,
    embed_dim_ratio=32,
    dropout=0.0,
):
    """
    Build PoseFormerV2 model architecture.

    This configuration must match the downloaded pretrained checkpoint.

    For checkpoint: 1_3_27_48.7.bin
        sequence_length = 27
        frame_kept = 1
        coeff_kept = 3
        depth = 4
        embed_dim_ratio = 32
    """

    args = SimpleNamespace(
        number_of_frames=sequence_length,
        number_of_kept_frames=frame_kept,
        number_of_kept_coeffs=coeff_kept,
        depth=depth,
        embed_dim_ratio=embed_dim_ratio,
    )

    model = PoseTransformerV2(
        num_frame=sequence_length,
        num_joints=17,
        in_chans=2,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=dropout,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        args=args,
    )

    return model


def load_poseformerv2_checkpoint(
    checkpoint_path,
    device=None,
    sequence_length=27,
    frame_kept=1,
    coeff_kept=3,
):
    """
    Load PoseFormerV2 pretrained checkpoint.

    Input:
        checkpoint_path: path to .bin checkpoint
        device: torch.device("cuda") or torch.device("cpu")

    Output:
        model: PoseFormerV2 model ready for inference
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_poseformerv2_model(
        sequence_length=sequence_length,
        frame_kept=frame_kept,
        coeff_kept=coeff_kept,
        depth=4,
        embed_dim_ratio=32,
        dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_pos" in checkpoint:
            state_dict = checkpoint["model_pos"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing_keys:
        print("WARNING: Missing keys:")
        for key in missing_keys[:20]:
            print("  ", key)
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")

    if unexpected_keys:
        print("WARNING: Unexpected keys:")
        for key in unexpected_keys[:20]:
            print("  ", key)
        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")

    model.to(device)
    model.eval()

    print("Loaded PoseFormerV2 checkpoint:")
    print("  Path:", checkpoint_path)
    print("  Device:", device)
    print("  Sequence length:", sequence_length)
    print("  Frame kept:", frame_kept)
    print("  Coeff kept:", coeff_kept)

    return model