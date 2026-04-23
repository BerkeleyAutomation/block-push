"""
run_push.py — hardcoded gripper-touches-cube test with recording.

Outputs:
  thirdview.mp4  — 480×480 third-view camera video
  attention.mp4  — 20×20 pixel attention window cropped from the thirdview frame

Usage:
    uv run python run_push.py
"""

import numpy as np
import imageio

from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z

# Recording settings
FPS = 20
ATTENTION_HALF = 10  # produces a 20×20 window


def step_toward(env, obs, target_pos, gripper=-1.0, max_steps=80, tol=0.005):
    """
    Move the end-effector toward target_pos using OSC position control.
    Yields obs after each step so the caller can record frames.
    """
    for _ in range(max_steps):
        eef_pos = obs["robot0_eef_pos"]
        err = target_pos - eef_pos
        dist = np.linalg.norm(err)
        if dist < tol:
            break
        action = np.zeros(7)
        # Scale error into [-1, 1]; OSC maps 1.0 → 0.05 m/step
        action[:3] = np.clip(err / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        yield obs


def hold(env, obs, n_steps, gripper=-1.0):
    """Hold current position for n_steps (zero action)."""
    for _ in range(n_steps):
        action = np.zeros(7)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        yield obs


def run_episode(env):
    """Execute the full scripted push and return (thirdview_frames, attention_frames)."""
    obs = env.reset()

    # ------------------------------------------------------------------
    # Fix attention window at cube-table contact point
    # ------------------------------------------------------------------
    cube_pos = env.get_cube_pos()
    contact_3d = np.array([cube_pos[0], cube_pos[1], TABLE_SURFACE_Z])
    u0, v0 = env.world_to_pixel(contact_3d)
    v0 += 99  # shift so cube bottom edge sits in top half of window

    print(f"Cube initial position : {cube_pos}")
    print(f"Attention window center: ({u0}, {v0}) px  [fixed for entire episode]")

    thirdview_frames = []
    attention_frames = []

    def record(obs):
        frame = env.get_frame(obs)  # (H, W, 3) uint8 RGB
        thirdview_frames.append(frame)
        H, W = frame.shape[:2]
        # Clamp window to image bounds
        r0 = max(0, v0 - ATTENTION_HALF)
        r1 = min(H, v0 + ATTENTION_HALF)
        c0 = max(0, u0 - ATTENTION_HALF)
        c1 = min(W, u0 + ATTENTION_HALF)
        crop = frame[r0:r1, c0:c1]
        # Pad if near edge to always output 20×20
        pad_h = (2 * ATTENTION_HALF) - crop.shape[0]
        pad_w = (2 * ATTENTION_HALF) - crop.shape[1]
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)))
        attention_frames.append(crop)

    # Record starting frame
    record(obs)

    # ------------------------------------------------------------------
    # Scripted push phases — gripper descends onto cube top then pushes forward
    # ------------------------------------------------------------------
    # Phase A: Move directly above the cube center
    print("\nPhase A: moving above cube top...")
    above = cube_pos + np.array([0.0, 0.0, 0.20])
    for obs in step_toward(env, obs, above, gripper=1.0, max_steps=100):
        record(obs)

    # Phase B: Descend to contact the cube top face
    print("Phase B: descending to touch cube top face...")
    touch = cube_pos + np.array([0.0, 0.0, 0.02])
    for obs in step_toward(env, obs, touch, gripper=-1.0, max_steps=120):
        record(obs)

    # Phase C: Push cube forward (+Y) while staying at the same height
    print("Phase C: pushing cube forward from top...")
    push_end = cube_pos + np.array([0.0, 0.05, 0.02])                                                                                        
    for obs in step_toward(env, obs, push_end, gripper=-1.0, max_steps=120, tol=0.01):
        record(obs)

    # Phase D: Retract upward
    print("Phase D: retracting arm...")
    eef_pos = obs["robot0_eef_pos"]
    retract = eef_pos + np.array([0.0, 0.0, 0.20])
    for obs in step_toward(env, obs, retract, gripper=1.0, max_steps=60):
        record(obs)

    final_cube_pos = env.get_cube_pos()
    displacement = final_cube_pos[1] - cube_pos[1]
    print(f"\nCube displaced {displacement:.3f} m in +Y.")
    print(f"Total frames recorded: {len(thirdview_frames)}")

    return thirdview_frames, attention_frames


def save_video(frames, path, fps, upscale=1):
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for frame in frames:
        if upscale > 1:
            frame = np.repeat(np.repeat(frame, upscale, axis=0), upscale, axis=1)
        writer.append_data(frame)
    writer.close()
    h, w = frames[0].shape[:2]
    print(f"Saved {path}  ({len(frames)} frames @ {fps} fps, {w*upscale}×{h*upscale} px)")


def main():
    print("Creating FrankaCubePushEnv...")
    env = FrankaCubePushEnv(
        camera_height=480,
        camera_width=480,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        has_renderer=False,
        horizon=2000,
        ignore_done=True,
    )

    thirdview_frames, attention_frames = run_episode(env)
    env.close()

    print("\nSaving videos...")
    save_video(thirdview_frames, "thirdview.mp4", FPS)
    save_video(attention_frames, "attention.mp4", FPS, upscale=20)
    print("\nDone. Output files: thirdview.mp4, attention.mp4")


if __name__ == "__main__":
    main()
