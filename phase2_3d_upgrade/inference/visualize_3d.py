import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Human3.6M-style skeleton
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),             # right leg
    (0, 4), (4, 5), (5, 6),             # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),    # spine/head
    (8, 11), (11, 12), (12, 13),        # left arm
    (8, 14), (14, 15), (15, 16)         # right arm
]

RIGHT_BODY = {
    (0, 1), (1, 2), (2, 3),
    (8, 14), (14, 15), (15, 16)
}

LEFT_BODY = {
    (0, 4), (4, 5), (5, 6),
    (8, 11), (11, 12), (12, 13)
}


def create_dummy_3d_pose_from_2d(pose_2d):
    """
    Only used for testing visualization.
    This is not PoseFormerV2 output.
    """

    pose_2d = np.asarray(pose_2d, dtype=np.float32)

    if pose_2d.shape != (17, 2):
        raise ValueError(f"Expected pose_2d shape (17, 2), got {pose_2d.shape}")

    pose_3d = np.zeros((17, 3), dtype=np.float32)
    pose_3d[:, 0] = pose_2d[:, 0]
    pose_3d[:, 1] = pose_2d[:, 1]
    pose_3d[:, 2] = np.linspace(-0.2, 0.2, 17)

    return pose_3d


def poseformer_to_visual_pose(pose_3d):
    """
    Convert PoseFormerV2 raw 3D output into a display coordinate system.

    Goal:
    - Fix left/right mirroring.
    - Make the skeleton stand upright.
    - Make the view closer to the input video.
    - Keep the pose root-centered at pelvis.

    Important:
    PoseFormerV2 output is a normalized, camera-relative 3D pose.
    It is not the same as pixel coordinates in the original RGB video.
    """

    pose_3d = np.asarray(pose_3d, dtype=np.float32)

    if pose_3d.shape != (17, 3):
        raise ValueError(f"Expected pose_3d shape (17, 3), got {pose_3d.shape}")

    # Root-center at pelvis
    pose = pose_3d.copy()
    pose = pose - pose[0:1]

    visual = np.zeros_like(pose, dtype=np.float32)

    # Raw model axes:
    # pose[:, 0] = model X
    # pose[:, 1] = model Y
    # pose[:, 2] = model Z
    #
    # Display axes:
    # X = left-right
    # Y = depth
    # Z = height
    #
    # The minus on X fixes mirror effect.
    visual[:, 0] = -pose[:, 0]
    visual[:, 1] = pose[:, 2]
    visual[:, 2] = -pose[:, 1]

    # If head is below pelvis, flip height.
    # Joint 10 = head, joint 0 = pelvis.
    if visual[10, 2] < visual[0, 2]:
        visual[:, 2] *= -1

    return visual


def set_equal_3d_axes(ax, pose_3d, radius=None):
    """
    Keep X/Y/Z scale equal so the skeleton is not stretched or squeezed.
    """

    xs = pose_3d[:, 0]
    ys = pose_3d[:, 1]
    zs = pose_3d[:, 2]

    if radius is None:
        max_range = np.array([
            xs.max() - xs.min(),
            ys.max() - ys.min(),
            zs.max() - zs.min()
        ]).max()

        radius = max(max_range * 0.70, 0.45)

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - radius, mid_x + radius)
    ax.set_ylim(mid_y - radius, mid_y + radius)
    ax.set_zlim(mid_z - radius, mid_z + radius)


def draw_3d_skeleton(ax, visual_pose):
    """
    Draw Human3.6M skeleton with red/blue body sides.
    """

    ax.scatter(
        visual_pose[:, 0],
        visual_pose[:, 1],
        visual_pose[:, 2],
        s=18,
        c="black",
        depthshade=False
    )

    for joint_a, joint_b in H36M_SKELETON:
        x_line = [visual_pose[joint_a, 0], visual_pose[joint_b, 0]]
        y_line = [visual_pose[joint_a, 1], visual_pose[joint_b, 1]]
        z_line = [visual_pose[joint_a, 2], visual_pose[joint_b, 2]]

        bone = (joint_a, joint_b)

        if bone in LEFT_BODY:
            color = "red"
        elif bone in RIGHT_BODY:
            color = "blue"
        else:
            color = "black"

        ax.plot(
            x_line,
            y_line,
            z_line,
            linewidth=3,
            color=color
        )


def save_3d_pose_image(pose_3d, output_path, title="PoseFormerV2 3D Pose"):
    """
    Save one 3D skeleton image using matplotlib.
    """

    pose_3d = np.asarray(pose_3d, dtype=np.float32)

    if pose_3d.shape != (17, 3):
        raise ValueError(f"Expected pose_3d shape (17, 3), got {pose_3d.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    visual_pose = poseformer_to_visual_pose(pose_3d)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    draw_3d_skeleton(ax, visual_pose)
    set_equal_3d_axes(ax, visual_pose)

    # Camera angle:
    # azim=-90 makes the 3D skeleton face closer to the 2D input view.
    # If later you still see left/right reversed, change azim to 90.
    ax.view_init(elev=12, azim=90)

    ax.set_title(title, pad=4)

    # Hide numeric labels to look closer to GitHub demo.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.grid(True)

    # Do not use tight_layout here because it spams warnings with 3D axes.
    plt.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=0.94)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def render_3d_pose_to_image_array(pose_3d, width=480, height=480):
    """
    Render 3D pose into OpenCV BGR image.
    Used when creating demo video or showing 3D in GUI.
    """

    temp_path = "phase2_3d_upgrade/outputs/demo_3d_frames/temp_3d_pose.png"

    save_3d_pose_image(
        pose_3d,
        temp_path,
        title="3D Pose"
    )

    image = cv2.imread(temp_path)

    if image is None:
        raise RuntimeError("Failed to render 3D pose image.")

    image = cv2.resize(image, (width, height))

    return image