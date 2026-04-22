"""
visualize_setup.py — verify the cube-push scene geometry before running the push.

Opens a Viser 3D viewer showing the table, green cube, and camera frustum.
Also saves setup_check.png with the attention window rectangle drawn on a rendered frame.

Usage:
    uv run python visualize_setup.py
Then open http://localhost:8080 in a browser.
"""

import sys
import time
import numpy as np
import cv2
import viser
import viser.transforms as tf

from cube_push_env import (
    FrankaCubePushEnv,
    CAMERA_POS,
    CAMERA_TARGET,
    CAMERA_FOVY,
    TABLE_SIZE,
    TABLE_OFFSET,
    TABLE_SURFACE_Z,
    CUBE_HALF,
    CUBE_POS,
)


def make_frustum_lines(cam_pos, cam_R, fovy_deg, aspect, depth=0.25):
    """Return line segments (pairs of points) forming a camera frustum wireframe."""
    fovy = np.deg2rad(fovy_deg)
    half_h = np.tan(fovy / 2) * depth
    half_w = half_h * aspect

    # Frustum corners in camera frame (camera looks in -Z)
    corners_cam = np.array([
        [ half_w,  half_h, -depth],
        [-half_w,  half_h, -depth],
        [-half_w, -half_h, -depth],
        [ half_w, -half_h, -depth],
    ])

    # Transform to world frame
    corners_world = (cam_R @ corners_cam.T).T + cam_pos

    lines = []
    # Four edges from apex to corners
    for c in corners_world:
        lines.append((cam_pos, c))
    # Four edges connecting corners
    for i in range(4):
        lines.append((corners_world[i], corners_world[(i + 1) % 4]))
    return lines


def main():
    print("Creating environment...")
    env = FrankaCubePushEnv(
        camera_height=480,
        camera_width=480,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        has_renderer=False,
    )
    obs = env.reset()

    # ------------------------------------------------------------------
    # Compute attention window center (fixed, cube-table contact point)
    # ------------------------------------------------------------------
    cube_pos = env.get_cube_pos()
    contact_3d = np.array([cube_pos[0], cube_pos[1], TABLE_SURFACE_Z])
    u0, v0 = env.world_to_pixel(contact_3d)
    half = 10  # 20×20 window

    print(f"\nScene info:")
    print(f"  Cube world position : {cube_pos}")
    print(f"  Camera position     : {CAMERA_POS}")
    print(f"  Camera target       : {CAMERA_TARGET}")
    print(f"  Contact point 3D    : {contact_3d}")
    print(f"  Attention window    : center=({u0}, {v0}), size=20×20 px")

    # ------------------------------------------------------------------
    # Save annotated frame
    # ------------------------------------------------------------------
    frame = env.get_frame(obs)  # (H, W, 3) RGB uint8
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw attention window rectangle
    x1, y1 = u0 - half, v0 - half
    x2, y2 = u0 + half, v0 + half
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.putText(frame_bgr, "Attention Window", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite("setup_check.png", frame_bgr)
    print("\nSaved setup_check.png with attention window drawn.")

    # ------------------------------------------------------------------
    # Viser 3D viewer
    # ------------------------------------------------------------------
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    time.sleep(0.5)  # let server start

    # World axes
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.003)

    # Table
    table_x, table_y, table_z_half = TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2] / 2
    server.scene.add_box(
        "/table",
        dimensions=(TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2]),
        position=(TABLE_OFFSET[0], TABLE_OFFSET[1], TABLE_OFFSET[2] - table_z_half),
        color=(220, 220, 220),
    )

    # Green cube
    server.scene.add_box(
        "/cube_green",
        dimensions=(CUBE_HALF * 2, CUBE_HALF * 2, CUBE_HALF * 2),
        position=tuple(CUBE_POS),
        color=(0, 200, 0),
    )

    # Contact point (attention anchor)
    server.scene.add_box(
        "/attention_anchor",
        dimensions=(0.01, 0.01, 0.01),
        position=tuple(contact_3d),
        color=(255, 0, 0),
    )

    # Camera position marker
    server.scene.add_box(
        "/camera_marker",
        dimensions=(0.04, 0.04, 0.04),
        position=tuple(CAMERA_POS),
        color=(80, 80, 80),
    )

    # Camera frustum wireframe (line segments via frame + axes)
    cam_pos_world, cam_R = env.get_camera_extrinsics()
    frustum_lines = make_frustum_lines(cam_pos_world, cam_R, CAMERA_FOVY, aspect=1.0)
    for i, (p0, p1) in enumerate(frustum_lines):
        mid = (p0 + p1) / 2
        direction = p1 - p0
        length = np.linalg.norm(direction)
        server.scene.add_box(
            f"/frustum/seg_{i}",
            dimensions=(0.003, 0.003, length),
            position=tuple(mid),
            color=(100, 100, 100),
        )

    # Line from camera to target
    server.scene.add_box(
        "/camera_line",
        dimensions=(0.005, 0.005, np.linalg.norm(CAMERA_TARGET - CAMERA_POS)),
        position=tuple((CAMERA_POS + CAMERA_TARGET) / 2),
        color=(150, 150, 255),
    )

    print("\nViser viewer running at http://localhost:8080")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()
