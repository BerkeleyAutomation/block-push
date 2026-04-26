"""
run_push.py — SAM3 + visual servoing cube push (no 3D cube coords).

Pipeline per episode:
  1. Reset env, capture first frame from thirdview + attention cameras
  2. SAM3 on attention frame → cube mask + centroid pixel
  3. Unproject centroid pixel to a 3D world point on the cube-center Z plane
  4. Phase A : move EEF to APPROACH_HEIGHT above the cube
  5. Phase B1: align EEF world XY with the cube world XY at current Z
  6. Phase B2: descend Z (action saturated) until stall- or flow-based contact
  7. Phase C : push +Y to displace the cube
  8. Phase D : retract straight up

Outputs (per invocation, written to results/run_<N>/):
  thirdview.mp4         — third-view recording with SAM3/EEF overlays
  attention.mp4         — crop of the attention-camera view (gripper-free)
  debug_first_frame.png — raw thirdview at start of episode
  debug_attn_frame.png  — raw attention frame at start of episode
  debug_sam3_attn.png   — SAM3 mask drawn over attention frame
  debug_sam3.png        — projected cube keypoints on thirdview

Usage:
    uv run python run_push.py [--checkpoint /path/to/sam3.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import imageio
import cv2

from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z, CUBE_HALF
from segmentation import load_sam3_model, segment_cube_sam3
from servoing import pixel_error_to_robot_delta, pixel_to_world_at_z

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Recording / window constants
# ---------------------------------------------------------------------------
FPS = 20
ATTENTION_HALF = 10   # 20×20 thirdview overlay rectangle (visualization only)
ATTN_RECORD_HALF = 60 # half-size of crop on the attention camera (gripper-free view)
ATTN_UPSCALE = 4      # upscale factor for the saved attention video
DETECT_HALF = 40      # 80×80 crop used for contact detection

# ---------------------------------------------------------------------------
# Servoing parameters
# ---------------------------------------------------------------------------
APPROACH_HEIGHT = 0.20       # m above cube top (Phase A target Z)
SERVO_KP = 1.0               # proportional XY gain; reduce if oscillating
STOP_PIXEL_THRESHOLD = 20    # px: Phase B1 done when |error| < this
DESCENT_ACTION_MAX = 0.15    # max |action[2]| in Phase B2 (≈7.5 mm/step OSC delta)
DESCENT_DEEP_Z = TABLE_SURFACE_Z - 0.10   # well below the table, drives action saturation
STALL_WINDOW = 15            # iterations to look back for stall detection
STALL_DZ_THRESH = 0.0008     # m: <0.8 mm drop in window → contact stall
CUBE_CENTER_Z = TABLE_SURFACE_Z + CUBE_HALF        # mass-centroid Z of the cube
CUBE_TOP_Z    = TABLE_SURFACE_Z + 2 * CUBE_HALF    # top face of the cube


# ---------------------------------------------------------------------------
# Contact detection  (unchanged)
# ---------------------------------------------------------------------------

def cube_contact_detection(prev_frames, current_gray, state,
                           threshold=15,
                           threshold_area=30,
                           stability_threshold=2,
                           flow_threshold=2.0):
    """
    Detect whether the cube has moved inside the attention window.
    Uses frame-difference + Farneback optical flow.
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
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag[mag < flow_threshold] = 0

    moving_pixels = np.sum(mag > flow_threshold)
    ratio = moving_pixels / (mag.size + 1e-5)
    mean_mag = np.mean(mag)
    max_mag = np.max(mag)

    print(f"[Contact] ratio={ratio:.3f}  mean_mag={mean_mag:.3f}  max_mag={max_mag:.3f}")

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
# Robot motion helpers  (unchanged)
# ---------------------------------------------------------------------------

def step_toward(env, obs, target_pos, gripper=-1.0, max_steps=80, tol=0.005):
    """Move EEF toward target_pos; yields obs after each step."""
    for _ in range(max_steps):
        eef_pos = obs["robot0_eef_pos"]
        err = target_pos - eef_pos
        if np.linalg.norm(err) < tol:
            break
        action = np.zeros(7)
        action[:3] = np.clip(err / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        yield obs


def hold(env, obs, n_steps, gripper=-1.0):
    """Hold current position for n_steps."""
    for _ in range(n_steps):
        action = np.zeros(7)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        yield obs


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def draw_debug_overlay(frame, mask, centroid_px, attn_anchor_px, eef_px=None):
    """Overlay keypoints, attention rectangle, and EEF dot (no mask fill in video)."""
    vis = frame.copy()

    # Centroid dot (red)
    if centroid_px is not None:
        cv2.circle(vis, centroid_px, 5, (255, 50, 50), -1)

    # Attention window rectangle (green)
    if attn_anchor_px is not None:
        ax, ay = attn_anchor_px
        cv2.rectangle(
            vis,
            (ax - ATTENTION_HALF, ay - ATTENTION_HALF),
            (ax + ATTENTION_HALF, ay + ATTENTION_HALF),
            (0, 255, 0), 1,
        )

    # EEF projection dot (yellow)
    if eef_px is not None:
        cv2.circle(vis, eef_px, 5, (255, 255, 0), -1)

    return vis


def save_sam3_debug(frame, mask, centroid_px, attn_anchor_px, path="debug_sam3.png"):
    """Save a static debug image showing the SAM3 detection (mask fill + keypoints)."""
    vis = frame.copy()
    if mask is not None:
        vis[mask] = (
            0.5 * vis[mask].astype(float) + 0.5 * np.array([80, 80, 255])
        ).astype(np.uint8)
        print(f"  SAM3 mask coverage: {mask.mean()*100:.1f}% of image")
    if centroid_px is not None:
        cv2.circle(vis, centroid_px, 6, (255, 50, 50), -1)
    if attn_anchor_px is not None:
        ax, ay = attn_anchor_px
        cv2.rectangle(vis,
                      (ax - ATTENTION_HALF, ay - ATTENTION_HALF),
                      (ax + ATTENTION_HALF, ay + ATTENTION_HALF),
                      (0, 255, 0), 2)
    # save as BGR for OpenCV
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  Saved SAM3 debug image: {path}")


# ---------------------------------------------------------------------------
# Main episode
# ---------------------------------------------------------------------------

def run_episode(env, processor, out_dir):
    """
    Execute the full SAM3-guided push and return (thirdview_frames, attention_frames).
    No 3D cube coordinates are used for navigation — all motion is derived from
    SAM3 detection + 2D pixel reprojection.

    Args:
        env:       FrankaCubePushEnv instance.
        processor: Sam3Processor returned by segmentation.load_sam3_model().
        out_dir:   pathlib.Path where debug PNGs are written.
    """
    obs = env.reset()

    # Let the simulation settle and the robot reach its home pose before we
    # capture the SAM3 frame — the arm is still moving immediately after reset.
    for obs in hold(env, obs, n_steps=20, gripper=1.0):
        pass

    first_frame = env.get_frame(obs)
    attn_frame = env.get_attention_frame(obs)

    # Save raw frames so you can inspect what SAM3 sees
    cv2.imwrite(str(out_dir / "debug_first_frame.png"), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "debug_attn_frame.png"),  cv2.cvtColor(attn_frame,  cv2.COLOR_RGB2BGR))
    print(f"Saved debug_first_frame.png and debug_attn_frame.png in {out_dir}")

    # ------------------------------------------------------------------
    # Step 1: SAM3 detection on attention frame (cube-only view, no arm)
    # ------------------------------------------------------------------
    print("\n--- SAM3 detection (attention camera) ---")
    sam3_mask, attn_centroid_px, _ = segment_cube_sam3(attn_frame, processor)
    save_sam3_debug(attn_frame, sam3_mask, attn_centroid_px, None,
                    str(out_dir / "debug_sam3_attn.png"))
    print("Saved debug_sam3_attn.png — open this to verify the cube was detected correctly")

    # Unproject attention centroid to 3D world point at the cube CENTER plane
    # (SAM3's mass-centroid sits roughly at the cube's vertical center, not its top).
    attn_cam_pos, attn_cam_R, attn_f, W, H = env.get_attn_camera_params()
    cube_3d = pixel_to_world_at_z(
        attn_centroid_px[0], attn_centroid_px[1],
        CUBE_CENTER_Z,
        attn_cam_pos, attn_cam_R, attn_f, W, H,
    )
    print(f"  cube_3d from attn SAM3: {cube_3d.round(3)}")

    # === DIAGNOSTIC: compare reprojected cube_3d to the simulator's ground truth ===
    # NOT used for navigation — only to verify the SAM3 → unproject pipeline.
    actual_cube = env.get_cube_pos()
    actual_attn_px  = env.attn_world_to_pixel(actual_cube)
    actual_third_px = env.world_to_pixel(actual_cube)
    print(f"  [diag] actual cube 3D       : {actual_cube.round(3)}")
    print(f"  [diag] actual cube attn px  : {actual_attn_px}    (SAM3 said {attn_centroid_px})")
    print(f"  [diag] actual cube third px : {actual_third_px}")
    print(f"  [diag] cube_3d - actual XY  : "
          f"({cube_3d[0]-actual_cube[0]:+.3f}, {cube_3d[1]-actual_cube[1]:+.3f})")

    # Project cube_3d to thirdview for overlay and contact detection crop
    cam_pos, cam_R, f, W, H = env.get_camera_params()
    centroid_px = env.world_to_pixel(cube_3d)
    attn_anchor_px = env.world_to_pixel(
        np.array([cube_3d[0], cube_3d[1], TABLE_SURFACE_Z])
    )
    print(f"  centroid_px (thirdview)={centroid_px}  attn_anchor_px={attn_anchor_px}")

    # Save thirdview debug with projected keypoints
    save_sam3_debug(first_frame, None, centroid_px, attn_anchor_px,
                    str(out_dir / "debug_sam3.png"))
    print("Saved debug_sam3.png — shows projected cube position on thirdview")

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    thirdview_frames = []
    attention_frames = []
    ax, ay = attn_anchor_px
    cax, cay = attn_centroid_px  # cube pixel in the attention camera (gripper-free)

    def record(obs, eef_px=None):
        frame = env.get_frame(obs)
        vis = draw_debug_overlay(frame, sam3_mask, centroid_px, attn_anchor_px, eef_px)
        thirdview_frames.append(vis)

        attn_full = env.get_attention_frame(obs)
        aH, aW = attn_full.shape[:2]
        r0 = max(0, cay - ATTN_RECORD_HALF)
        r1 = min(aH, cay + ATTN_RECORD_HALF)
        c0 = max(0, cax - ATTN_RECORD_HALF)
        c1 = min(aW, cax + ATTN_RECORD_HALF)
        crop = attn_full[r0:r1, c0:c1]
        ph = (2 * ATTN_RECORD_HALF) - crop.shape[0]
        pw = (2 * ATTN_RECORD_HALF) - crop.shape[1]
        if ph > 0 or pw > 0:
            crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)))
        attention_frames.append(crop)
        return frame

    # Record first frame (SAM3 ran on this)
    record(obs)

    # Detection crop bounds (used for contact detection in Phase B2)
    dx1 = max(ax - DETECT_HALF, 0)
    dy1 = max(ay - DETECT_HALF, 0)
    dx2 = min(ax + DETECT_HALF, W)
    dy2 = min(ay + DETECT_HALF, H)

    # ------------------------------------------------------------------
    # Phase A: move above cube (cube_3d XY, approach height Z)
    # ------------------------------------------------------------------
    print("\n--- Phase A: move above cube ---")
    above_3d = np.array([cube_3d[0], cube_3d[1], CUBE_TOP_Z + APPROACH_HEIGHT])
    print(f"  cube_3d={cube_3d.round(3)}  above_3d={above_3d.round(3)}")
    for obs in step_toward(env, obs, above_3d, gripper=1.0, max_steps=100):
        eef_px = env.world_to_pixel(obs["robot0_eef_pos"])
        record(obs, eef_px)

    # ------------------------------------------------------------------
    # Phase B1: XY alignment — move EEF to cube XY at current Z
    # ------------------------------------------------------------------
    print("\n--- Phase B1: XY alignment ---")
    print(f"  cube world XY: ({cube_3d[0]:.3f}, {cube_3d[1]:.3f})")
    eef_z = obs["robot0_eef_pos"][2]
    b1_target = np.array([cube_3d[0], cube_3d[1], eef_z])
    for obs in step_toward(env, obs, b1_target, gripper=-1.0, max_steps=80, tol=0.003):
        eef_px = env.world_to_pixel(obs["robot0_eef_pos"])
        print(f"  EEF px={eef_px}  target_cube_px={centroid_px}")
        record(obs, eef_px)

    # ------------------------------------------------------------------
    # Phase B2: Z descent along cube XY column until contact.
    #
    # Strategy: target a deep Z below the table so action[2] saturates and the
    # OSC drives a steady descent.  Detect contact via Z stall (EEF stops
    # dropping despite commanding descent) — robust regardless of camera noise.
    # Also keep optical-flow detector as a secondary signal.
    # ------------------------------------------------------------------
    print("\n--- Phase B2: Z approach to cube ---")
    prev_detect_frames = []
    contact_state = {'contact_frame_count': 0}
    contact_confirmed = False
    z_history = []
    start_z = obs["robot0_eef_pos"][2]
    print(f"  start EEF z={start_z:.4f}, target deep z={DESCENT_DEEP_Z:.4f}")

    for step_i in range(500):
        eef_pos = obs["robot0_eef_pos"]
        target_3d = np.array([cube_3d[0], cube_3d[1], DESCENT_DEEP_Z])
        raw = (target_3d - eef_pos) / 0.05
        action = np.zeros(7)
        action[0] = float(np.clip(raw[0], -1.0, 1.0))
        action[1] = float(np.clip(raw[1], -1.0, 1.0))
        action[2] = float(np.clip(raw[2], -DESCENT_ACTION_MAX, DESCENT_ACTION_MAX))
        action[6] = -1.0
        obs, _, _, _ = env.step(action)
        frame = record(obs, env.world_to_pixel(obs["robot0_eef_pos"]))

        z_now = obs["robot0_eef_pos"][2]
        z_history.append(z_now)
        if len(z_history) > STALL_WINDOW:
            z_history.pop(0)

        if step_i % 10 == 0:
            print(f"  step={step_i:3d}  EEF z={z_now:.4f}")

        # Z-stall contact: EEF can't drop further despite commanding descent
        if len(z_history) >= STALL_WINDOW:
            dz_window = z_history[0] - z_history[-1]
            if dz_window < STALL_DZ_THRESH and z_now < start_z - 0.05:
                print(f"====== STALL CONTACT at step {step_i}  "
                      f"EEF z={z_now:.4f}  dz over {STALL_WINDOW}={dz_window:.5f} ======")
                contact_confirmed = True
                break

        # Optical-flow contact (secondary)
        crop_gray = cv2.cvtColor(
            cv2.cvtColor(frame[dy1:dy2, dx1:dx2], cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2GRAY,
        )
        prev_detect_frames.append(crop_gray)
        if len(prev_detect_frames) > 5:
            prev_detect_frames.pop(0)

        if len(prev_detect_frames) >= 4:
            flow_contact, _ = cube_contact_detection(
                prev_detect_frames[:-1], crop_gray, contact_state)
            if flow_contact:
                print(f"====== FLOW CONTACT at step {step_i}  EEF z={z_now:.4f} ======")
                contact_confirmed = True
                break

    if not contact_confirmed:
        print(f"[Phase B2] Max iterations reached. Final EEF z={obs['robot0_eef_pos'][2]:.4f}")

    # ------------------------------------------------------------------
    # Phase C: push +Y from current EEF position
    # ------------------------------------------------------------------
    print("\n--- Phase C: push cube ---")
    cube_before = env.get_cube_pos()
    eef_pos = obs["robot0_eef_pos"]
    push_target = eef_pos + np.array([0.0, 0.05, 0.0])
    for obs in step_toward(env, obs, push_target, gripper=-1.0, max_steps=120, tol=0.01):
        record(obs)
    cube_after = env.get_cube_pos()
    print(f"  cube before push: {cube_before.round(3)}")
    print(f"  cube after  push: {cube_after.round(3)}")
    print(f"  cube delta XYZ  : {(cube_after - cube_before).round(3)}")

    # ------------------------------------------------------------------
    # Phase D: retract
    # ------------------------------------------------------------------
    print("\n--- Phase D: retract ---")
    eef_pos = obs["robot0_eef_pos"]
    retract = eef_pos + np.array([0.0, 0.0, 0.20])
    for obs in step_toward(env, obs, retract, gripper=1.0, max_steps=60):
        record(obs)

    print(f"\nTotal frames: {len(thirdview_frames)}")
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
    print(f"Saved {path}  ({len(frames)} frames @ {fps} fps, {w * upscale}×{h * upscale})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _next_run_dir() -> Path:
    """Return results/run_<N>/ with N = max existing index + 1 (1-based)."""
    RESULTS_ROOT.mkdir(exist_ok=True)
    existing = [int(p.name.split("_")[1]) for p in RESULTS_ROOT.glob("run_*")
                if p.is_dir() and p.name.split("_", 1)[1].isdigit()]
    n = (max(existing) + 1) if existing else 1
    out = RESULTS_ROOT / f"run_{n}"
    out.mkdir()
    return out


def main():
    parser = argparse.ArgumentParser(description="SAM3 + visual servoing cube push")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Local SAM3 checkpoint (.pt). Omit to load from HuggingFace.",
    )
    args = parser.parse_args()

    out_dir = _next_run_dir()
    print(f"Run output directory: {out_dir}\n")

    print("Loading SAM3 model...")
    _, processor = load_sam3_model(args.checkpoint)
    print("SAM3 loaded.\n")

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

    thirdview_frames, attention_frames = run_episode(env, processor, out_dir)
    env.close()

    print("\nSaving videos...")
    save_video(thirdview_frames, str(out_dir / "thirdview.mp4"), FPS)
    save_video(attention_frames, str(out_dir / "attention.mp4"), FPS, upscale=ATTN_UPSCALE)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
