"""
servoing.py — pixel-space to robot-space math for visual servoing.

All functions work with the coordinate conventions of FrankaCubePushEnv:
  - Camera looks along its -Z axis (points in front have p_cam[2] < 0)
  - world_to_pixel: u = -f*p_cam[0]/p_cam[2] + W/2
                    v = H - 1 - (f*p_cam[1]/p_cam[2] + H/2)
"""

import numpy as np


def pixel_error_to_robot_delta(e_u, e_v, cam_pos, cam_R, eef_pos, f, Kp=1.0):
    """
    Convert 2D pixel error (target - current) to world-frame XY delta.

    Builds the full 2x2 image Jacobian J = d(u,v)/d(world_x, world_y) and
    solves J @ [dx, dy] = [e_u, e_v].  The naive depth-only formula fails for
    tilted cameras because world-Y movement also changes the camera depth
    p_cam[2], so both projection terms contribute.

    Derivation (camera looks along -Z; pixel u, v are in screen convention,
    where v_screen = H-1-v_gl with v_gl = -f*p_cam[1]/depth + H/2):
      du   = -f*dp[0]/depth + f*p[0]*dp[2]/depth^2     # u == u_gl
      dv_s =  f*dp[1]/depth - f*p[1]*dp[2]/depth^2     # screen v
    where dp = cam_R.T @ dP_world, dP_world = [dx, dy, 0].

    Args:
        e_u, e_v:  pixel error = target_pixel - current_eef_pixel
        cam_pos:   (3,) camera position in world frame
        cam_R:     (3,3) camera-to-world rotation (columns: right, up, -forward)
        eef_pos:   (3,) current EEF world position
        f:         focal length in pixels
        Kp:        proportional gain (reduce if oscillating)

    Returns:
        (3,) world-frame delta [dx, dy, 0]
    """
    p_cam = cam_R.T @ (eef_pos - cam_pos)
    depth = p_cam[2]  # negative for points in front of camera

    # Build 2x2 image Jacobian: J[r, i] = d(pixel_r) / d(world_i), i in {x, y}
    R_inv = cam_R.T  # world-to-camera rotation
    J = np.zeros((2, 2))
    for i in range(2):  # i=0 -> world X, i=1 -> world Y
        J[0, i] = (-f * R_inv[0, i] / depth
                   + f * p_cam[0] * R_inv[2, i] / depth**2)
        # Screen-v gradient has opposite sign from v_gl gradient.
        J[1, i] = ( f * R_inv[1, i] / depth
                   - f * p_cam[1] * R_inv[2, i] / depth**2)

    try:
        delta_xy = np.linalg.solve(J, np.array([float(e_u), float(e_v)]))
    except np.linalg.LinAlgError:
        return np.zeros(3)

    return Kp * np.array([delta_xy[0], delta_xy[1], 0.0])


def pixel_to_world_at_z(u, v, z_world, cam_pos, cam_R, f, W, H):
    """
    Unproject screen pixel (u, v) to the 3D world point at height z_world.

    Exact inverse of FrankaCubePushEnv.world_to_pixel().

    Args:
        u, v:      integer pixel coordinates (top-left origin, v increases down)
        z_world:   target world Z plane to intersect
        cam_pos:   (3,) camera position in world frame
        cam_R:     (3,3) camera-to-world rotation
        f:         focal length in pixels
        W, H:      image width and height in pixels

    Returns:
        (3,) world point on the plane z = z_world whose projection is (u, v)
    """
    v_gl = H - 1 - v  # flip to OpenGL convention (origin at bottom-left)

    # Ray direction in camera frame (unnormalised; depth is the free parameter).
    # Inverts world_to_pixel: u_gl = -f*p[0]/d + W/2, v_gl = -f*p[1]/d + H/2.
    ax = -(u - W / 2.0) / f       # p_cam[0] = ax * depth
    ay = -(v_gl - H / 2.0) / f    # p_cam[1] = ay * depth

    # Solve for depth so that the world Z coordinate equals z_world:
    # cam_pos[2] + (cam_R @ [ax*d, ay*d, d])[2] = z_world
    denom = cam_R[2, 0] * ax + cam_R[2, 1] * ay + cam_R[2, 2]
    depth = (z_world - cam_pos[2]) / denom

    p_cam = np.array([ax * depth, ay * depth, depth])
    return cam_pos + cam_R @ p_cam
