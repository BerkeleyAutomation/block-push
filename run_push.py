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
import cv2

from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z

# Recording settings
FPS = 20
ATTENTION_HALF = 10  # produces a 20×20 window
ATTN_V_OFFSET = 99   # pixels below the projected contact point; tunes vertical placement

# Detection window (larger than recording window)
DETECT_HALF = 40     # produces an 80×80 detection crop


# ---------------------------------------------------------------------------
# Cube segmentation
# ---------------------------------------------------------------------------

def segment_cube_canny(image, attn_px, window_size=DETECT_HALF):
    """
    Canny edge + largest-contour fill to segment the cube inside the
    attention window.

    Args:
        image:       RGB uint8 (H, W, 3) from env.get_frame()
        attn_px:     (x, y) pixel centre of attention window
        window_size: half-side of the square crop (pixels)

    Returns:
        binary_mask: uint8 (H, W) — cube region = 255, everything else = 0
    """
    x, y = attn_px
    H, W = image.shape[:2]
    x1 = max(x - window_size, 0)
    y1 = max(y - window_size, 0)
    x2 = min(x + window_size, W)
    y2 = min(y + window_size, H)

    crop_bgr = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cube_mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(cube_mask, [largest], -1, 255, -1)

    binary_mask = np.zeros((H, W), dtype=np.uint8)
    binary_mask[y1:y2, x1:x2] = cube_mask
    return binary_mask


# ---------------------------------------------------------------------------
# Motion / contact detection
# ---------------------------------------------------------------------------

def cube_contact_detection(prev_frames, current_gray, state,
                           threshold=15,
                           threshold_area=30,
                           stability_threshold=2,
                           flow_threshold=2.0):
    """
    Detect whether the cube has moved inside the attention window.
    Uses frame-difference + Farneback optical flow.

    Args:
        prev_frames:        list of recent grayscale crops (at least 3)
        current_gray:       current grayscale attention-window crop
        state:              dict with key 'contact_frame_count' (persists across calls)
        threshold:          pixel-intensity change threshold for diff binarisation
        threshold_area:     minimum contour area (px²) to count as motion
        stability_threshold: consecutive motion frames required before contact fires
        flow_threshold:     optical-flow magnitude below which motion is ignored

    Returns:
        contact_detected: bool
        img_vis:          BGR debug image (prev | current | diff | flow)
    """
    if len(prev_frames) < 3:
        return False, cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)

    prev_gray = np.median(np.array(prev_frames), axis=0).astype(np.uint8)

    delta = cv2.absdiff(prev_gray, current_gray)
    delta_blur = cv2.GaussianBlur(delta, (5, 5), 0)
    _, thresh = cv2.threshold(delta_blur, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > threshold_area]

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag[mag < flow_threshold] = 0

    moving_pixels = np.sum(mag > flow_threshold)
    ratio = moving_pixels / (mag.size + 1e-5)
    mean_mag = np.mean(mag)
    max_mag = np.max(mag)

    print(f"[Cube Contact] ratio={ratio:.3f}  mean_mag={mean_mag:.3f}  max_mag={max_mag:.3f}")

    motion_detected = bool(large_contours and ratio > 0.15 and mean_mag > 0.4)

    if motion_detected:
        state['contact_frame_count'] += 1
        contact_detected = state['contact_frame_count'] >= stability_threshold
        if contact_detected:
            state['contact_frame_count'] = 0
    else:
        state['contact_frame_count'] = 0
        contact_detected = False

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_vis = np.hstack([
        cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET),
    ])

    return contact_detected, img_vis


# ---------------------------------------------------------------------------
# Robot motion helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main episode
# ---------------------------------------------------------------------------

def run_episode(env):
    """Execute the full scripted push and return (thirdview_frames, attention_frames)."""
    obs = env.reset()

    # ------------------------------------------------------------------
    # Fix attention window at cube-table contact point
    # ------------------------------------------------------------------
    cube_pos = env.get_cube_pos()
    contact_3d = np.array([cube_pos[0], cube_pos[1], TABLE_SURFACE_Z])
    u0, v0 = env.world_to_pixel(contact_3d)
    v0 += ATTN_V_OFFSET

    print(f"Cube initial position : {cube_pos}")
    print(f"Attention window centre: ({u0}, {v0}) px  [fixed for entire episode]")

    thirdview_frames = []
    attention_frames = []

    def record(obs):
        frame = env.get_frame(obs)  # (H, W, 3) uint8 RGB
        thirdview_frames.append(frame)
        H, W = frame.shape[:2]
        r0 = max(0, v0 - ATTENTION_HALF)
        r1 = min(H, v0 + ATTENTION_HALF)
        c0 = max(0, u0 - ATTENTION_HALF)
        c1 = min(W, u0 + ATTENTION_HALF)
        crop = frame[r0:r1, c0:c1]
        pad_h = (2 * ATTENTION_HALF) - crop.shape[0]
        pad_w = (2 * ATTENTION_HALF) - crop.shape[1]
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)))
        attention_frames.append(crop)
        return frame  # returned for use in detection

    # Record starting frame and derive image bounds for detection crop
    first_frame = record(obs)
    H_img, W_img = first_frame.shape[:2]
    dx1 = max(u0 - DETECT_HALF, 0)
    dy1 = max(v0 - DETECT_HALF, 0)
    dx2 = min(u0 + DETECT_HALF, W_img)
    dy2 = min(v0 + DETECT_HALF, H_img)

    # ------------------------------------------------------------------
    # Phase A: Move directly above the cube centre
    # ------------------------------------------------------------------
    print("\nPhase A: moving above cube top...")
    above = cube_pos + np.array([0.0, 0.0, 0.20])
    for obs in step_toward(env, obs, above, gripper=1.0, max_steps=100):
        record(obs)

    # ------------------------------------------------------------------
    # Phase B: Descend toward cube — stop on contact detection
    # ------------------------------------------------------------------
    print("Phase B: descending to touch cube top face...")
    touch = cube_pos + np.array([0.0, 0.0, 0.02])

    prev_detect_frames = []
    contact_state = {'contact_frame_count': 0}
    contact_confirmed = False

    for obs in step_toward(env, obs, touch, gripper=-1.0, max_steps=120):
        frame = record(obs)

        # 1. Segment cube inside detection window
        binary_mask = segment_cube_canny(frame, (u0, v0))

        # 2. Build masked grayscale crop (only cube pixels contribute)
        crop_gray = cv2.cvtColor(
            cv2.cvtColor(frame[dy1:dy2, dx1:dx2], cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2GRAY,
        )
        mask_crop = binary_mask[dy1:dy2, dx1:dx2]
        crop_gray = cv2.bitwise_and(crop_gray, crop_gray, mask=mask_crop)

        # 3. Maintain rolling frame buffer (last 5)
        prev_detect_frames.append(crop_gray)
        if len(prev_detect_frames) > 5:
            prev_detect_frames.pop(0)

        # 4. Run contact detection once prev_frames[:-1] has ≥3 frames
        if len(prev_detect_frames) >= 4:
            contact_confirmed, debug_vis = cube_contact_detection(
                prev_detect_frames[:-1], crop_gray, contact_state)

            if contact_confirmed:
                print("====== CONTACT DETECTED! Stopping robot ======")
                break

    if not contact_confirmed:
        print("[Phase B] Reached target without triggering contact detection.")

    # ------------------------------------------------------------------
    # Phase C: Push cube forward (+Y) while staying at the same height
    # ------------------------------------------------------------------
    print("Phase C: pushing cube forward from top...")
    push_end = cube_pos + np.array([0.0, 0.05, 0.02])
    for obs in step_toward(env, obs, push_end, gripper=-1.0, max_steps=120, tol=0.01):
        record(obs)

    # ------------------------------------------------------------------
    # Phase D: Retract upward
    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

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
