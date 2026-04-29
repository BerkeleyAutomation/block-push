"""
run_push.py — SAM3 + image-space visual servoing cube push (no cube depth).

Single-camera pipeline.  The 20×20 attention window is cropped from the
thirdview frame at the cube-table interface; gripper image is locked to
the cube pixel by Phase-2 ray-following so the gripper stays above the
window.

Pipeline per episode:
  1. Reset env, capture first thirdview frame
  2. SAM3 on thirdview frame → cube mask + centroid + bottom-edge anchor.
     Phases B1/B2 servo to the mask BOTTOM-edge midpoint (front-face-top
     of the cube) so the gripper tip lands on the front face of the cube;
     the centroid is shown only for diagnostics.
  3. Phase A : move EEF to APPROACH_HEIGHT above the table at current tip XY
  4. Phase B1: image-space XY alignment — at each step solve for the world XY
               (at the gripper tip's CURRENT Z) that projects to the cube's
               front-face-top pixel; step toward it until the tip pixel is
               within STOP_PIXEL_THRESHOLD
  5. Phase B2: descend along the thirdview camera ray (keeps front-face-top
               pixel locked) until the cube's yellow pixels in the 20×20 window
               crop drop below YELLOW_DROP_FRACTION of their initial count
               — i.e. the cube has translated up and out of the window.
               On the trigger, re-run SAM3 on the current thirdview frame
               and report the cube's 3D position from sim ground truth.

The cube's 3D height is never used for control — only the EEF's own
(kinematically known) depth and the camera intrinsics + extrinsics.  The
final 3D coords printed on contact are simulator ground truth, for
verification only.

Outputs (per invocation, written to results/run_<N>/):
  thirdview.mp4         — third-view recording with SAM3/EEF overlays
  attention.mp4         — 20×20 thirdview crop of the attention window,
                          upscaled 12×
  sideview.mp4          — perpendicular side-view camera (visual sanity
                          check that the gripper tip contacts the cube
                          front face at the correct height)
  debug_first_frame.png — raw thirdview at start of episode
  debug_sam3.png        — SAM3 mask + 20×20 attention window on thirdview

Usage:
    uv run python run_push.py [--checkpoint /path/to/sam3.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import imageio
import cv2

from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z
from segmentation import load_sam3_model, segment_cube_sam3
from servoing import pixel_to_world_at_z

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Recording / window constants
# ---------------------------------------------------------------------------
FPS = 20
ATTENTION_HALF = 6   # 20×20 thirdview overlay rectangle (visualization)
DETECT_HALF = 6      # 20×20 crop used for contact detection
# SAM3's mask on thirdview underestimates the cube (only catches the top
# face), so its bottom_y sits on the cube body rather than the actual
# cube-table line.  Push the anchor down by this many pixels so the
# 20×20 window straddles the real cube-bottom edge in image space — the
# upper half of the window contains cube body, the lower half contains
# the table strip just below the cube.  When the cube is pushed in +Y
# (away from camera), the cube image translates UP and exits the window
# from the top, leaving pure table content behind.
WINDOW_DROP_PX = 31  # tuned for CUBE_HALF=0.035 (7 cm cube)

# ---------------------------------------------------------------------------
# Servoing parameters — purely image-space alignment + camera-ray descent.
# Cube depth is NEVER assumed.
# ---------------------------------------------------------------------------
APPROACH_HEIGHT = 0.20       # m above table surface (Phase A target Z)
STOP_PIXEL_THRESHOLD = 20    # px: Phase B1 done when |EEF px - cube px| < this
SERVO_DZ = 0.008             # camera-frame depth step per Phase-2 iteration (m).
                             # World-Z descent is ~SERVO_DZ * cos(camera_tilt),
                             # and the OSC controller tracks at ~15% of command,
                             # so 0.008 → ~0.5 mm actual world-Z descent per step.
PHASE1_MAX_STEPS = 120
PHASE2_MAX_STEPS = 600
PHASE2_WARMUP_STEPS = 20     # skip checks only for early alignment transients;
                             # do not block stopping for most of the descent
ATTN_VIDEO_UPSCALE = 12      # upscale 20×20 attn crop → 240×240 in the saved video
# Detection: count near-white (table) pixels in the attention-window crop.
# The cube fills the upper portion of the window initially (cube body =
# coloured, not white), so initial white count is low.  When the gripper
# pushes the cube +Y (away from camera, up in image), the cube exits the
# window from above and only table remains — white count rises sharply.
# Trigger fires once the window is dominantly white.
WHITE_RGB_THRESH = 200           # per-channel: R, G, AND B must each exceed
                                 # this to be "white" (rejects cube body and
                                 # gripper, both of which fail the B channel)
WHITE_TRIGGER_FRACTION = 0.60    # trigger when white fraction of window
                                 # pixels exceeds this (cube is gone, table
                                 # fills the window)
WHITE_MIN_RISE = 20              # absolute minimum rise in white count from
                                 # initial — guards against false fires when
                                 # the initial frame already happened to have
                                 # mostly table (cube barely in window)


# ---------------------------------------------------------------------------
# Robot motion helpers  (unchanged)
# ---------------------------------------------------------------------------

def step_toward(env, obs, target_pos, gripper=-1.0, max_steps=80, tol=0.005):
    """Move gripper tip toward target_pos; yields obs after each step."""
    for _ in range(max_steps):
        tip_pos = env.get_gripper_tip_pos()
        err = target_pos - tip_pos
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

    # Save raw frame so you can inspect what SAM3 sees
    cv2.imwrite(str(out_dir / "debug_first_frame.png"),
                cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    print(f"Saved debug_first_frame.png in {out_dir}")

    # ------------------------------------------------------------------
    # SAM3 detection on the (only) thirdview frame.  Returns the cube
    # mask centroid (used for Phase-1 image-space servoing) and the
    # mask's bottom-edge midpoint, which we shift down by WINDOW_DROP_PX
    # to anchor the 20×20 attention window at the real cube-table line.
    # ------------------------------------------------------------------
    print("\n--- SAM3 detection (thirdview camera) ---")
    # The cube is small (~25 px) in the full thirdview frame — too small for
    # SAM3 to reliably detect.  Crop a generous image-space window over the
    # table region and upscale before running SAM3.  Center is a fixed
    # image-space prior (lower-center of the frame, where the table sits in
    # this fixed-camera setup); no cube 3D pose is used.
    H_img, W_img = first_frame.shape[:2]
    crop_center_px = (W_img // 2, int(H_img * 0.6))
    print(f"  SAM3 crop_center={crop_center_px}  half=120  upscale=4")
    sam3_mask, centroid_px, sam3_bottom_px = segment_cube_sam3(
        first_frame, processor,
        crop_center=crop_center_px, crop_half=120, upscale=4,
    )
    # Anchor the window at the cube-table line: SAM3 mask bottom (top-face
    # bottom edge in image) + WINDOW_DROP_PX to reach the actual cube body
    # bottom + a strip of table.  Cube is in the upper half of the window
    # initially; when the gripper pushes the cube +Y (away from camera, up
    # in image), the cube exits the window from above and only table
    # remains.
    attn_anchor_px = (sam3_bottom_px[0], sam3_bottom_px[1] + WINDOW_DROP_PX)
    print(f"  centroid_px={centroid_px}  sam3_bottom_px={sam3_bottom_px}  "
          f"attn_window_px={attn_anchor_px}")

    eef_ref = obs["robot0_eef_pos"]
    tip = env.get_gripper_tip_pos()
    print(f"  [diag] eef_ref={eef_ref.round(4)}  tip={tip.round(4)}  "
          f"delta={(tip - eef_ref).round(4)}")

    # === DIAGNOSTIC: compare SAM3 detection to ground-truth pixel projection ===
    actual_cube = env.get_cube_pos()
    actual_third_px = env.world_to_pixel(actual_cube)
    print(f"  [diag] actual cube 3D : {actual_cube.round(3)}")
    print(f"  [diag] actual cube px : {actual_third_px}    (SAM3 said {centroid_px})")
    print(f"  [diag] pixel error    : "
          f"({centroid_px[0]-actual_third_px[0]:+d}, {centroid_px[1]-actual_third_px[1]:+d})")

    save_sam3_debug(first_frame, sam3_mask, centroid_px, attn_anchor_px,
                    str(out_dir / "debug_sam3.png"))
    print("Saved debug_sam3.png — verify the 20×20 attention window sits on "
          "the cube-bottom + table strip in the thirdview frame")

    # Thirdview camera params (used by Phases B1 + B2)
    cam_pos, cam_R, f, W, H = env.get_camera_params()

    # 20×20 detection-crop bounds in the THIRDVIEW frame.
    awx, awy = attn_anchor_px
    adx1 = max(awx - DETECT_HALF, 0)
    ady1 = max(awy - DETECT_HALF, 0)
    adx2 = min(awx + DETECT_HALF, W)
    ady2 = min(awy + DETECT_HALF, H)

    # ------------------------------------------------------------------
    # Recording helpers — attention.mp4 shows ONLY the 20×20 thirdview
    # crop at the attention window (upscaled), so the user sees exactly
    # what the contact algorithm sees.  Thirdview overlay still draws
    # the full scene.
    # ------------------------------------------------------------------
    thirdview_frames = []
    attention_frames = []  # each frame is the 20×20 crop, upscaled at save
    sideview_frames = []   # perpendicular camera, for visual verification

    def record(obs, eef_px=None):
        frame = env.get_frame(obs)
        vis = draw_debug_overlay(frame, sam3_mask, centroid_px, attn_anchor_px, eef_px)
        thirdview_frames.append(vis)

        crop = frame[ady1:ady2, adx1:adx2]
        # Pad to consistent 20×20 if window clipped at frame edge
        ph = (2 * DETECT_HALF) - crop.shape[0]
        pw = (2 * DETECT_HALF) - crop.shape[1]
        if ph > 0 or pw > 0:
            crop = np.pad(crop, ((0, ph), (0, pw), (0, 0)))
        attention_frames.append(crop)

        sideview_frames.append(env.get_sideview_frame(obs))
        return frame

    record(obs)

    # ------------------------------------------------------------------
    # Phase A: move EEF straight up to APPROACH_HEIGHT above the table.
    # No cube coords used — we just lift to a known-safe height before servoing.
    # ------------------------------------------------------------------
    print("\n--- Phase A: lift to approach height ---")
    tip_now = env.get_gripper_tip_pos()
    above_3d = np.array([tip_now[0], tip_now[1], TABLE_SURFACE_Z + APPROACH_HEIGHT])
    print(f"  current tip={tip_now.round(3)}  above_target={above_3d.round(3)}")
    for obs in step_toward(env, obs, above_3d, gripper=1.0, max_steps=100):
        eef_px = env.world_to_pixel(env.get_gripper_tip_pos())
        record(obs, eef_px)

    # ------------------------------------------------------------------
    # Phase B1: image-space XY alignment (no cube depth).
    #
    # At each step, solve for the world XY at the EEF's CURRENT Z that
    # projects to the cube's pixel — that's the world point through the
    # camera ray at the gripper's known depth — and step toward it.
    # ------------------------------------------------------------------
    print("\n--- Phase B1: image-space XY alignment ---")
    for step_i in range(PHASE1_MAX_STEPS):
        tip_pos = env.get_gripper_tip_pos()
        tip_z = float(tip_pos[2])
        target_world = pixel_to_world_at_z(
            sam3_bottom_px[0], sam3_bottom_px[1], tip_z,
            cam_pos, cam_R, f, W, H,
        )
        target_world[2] = tip_z  # alignment is pure XY — preserve Z
        err = target_world - tip_pos
        action = np.zeros(7)
        action[:3] = np.clip(err / 0.05, -1.0, 1.0)
        action[6] = 1.0  # keep gripper closed so fingers stay at the tip site
        obs, _, _, _ = env.step(action)
        eef_px = env.world_to_pixel(env.get_gripper_tip_pos())
        record(obs, eef_px)

        pix_err = float(np.hypot(eef_px[0] - sam3_bottom_px[0],
                                 eef_px[1] - sam3_bottom_px[1]))
        if step_i % 5 == 0:
            print(f"  step={step_i:3d}  tip px={eef_px}  front-face px={sam3_bottom_px}  "
                  f"pix_err={pix_err:.1f}")
        if pix_err < STOP_PIXEL_THRESHOLD:
            print(f"====== ALIGNED at step {step_i}  pix_err={pix_err:.1f} ======")
            break
    else:
        print(f"[Phase B1] Max iterations reached.  pix_err={pix_err:.1f}")

    # ------------------------------------------------------------------
    # Phase B2: descend along the thirdview camera ray through the cube
    # pixel.  The cube pixel stays locked in the image while the EEF moves
    # toward it; stop when the 20×20 attention window detects motion via
    # image differencing + Farneback optical flow.
    # ------------------------------------------------------------------
    print("\n--- Phase B2: camera-ray descent ---")
    # Ray direction in camera frame for the front-face-top pixel.  Same
    # convention as pixel_to_world_at_z: p_cam[0] = ax_ray*z,
    # p_cam[1] = ay_ray*z, p_cam[2] = z.
    u_cube, v_cube = sam3_bottom_px
    v_gl = H - 1 - v_cube
    ax_ray = -(u_cube - W / 2.0) / f
    ay_ray = -(v_gl - H / 2.0) / f

    contact_confirmed = False
    start_z = float(env.get_gripper_tip_pos()[2])
    print(f"  start tip z={start_z:.4f}, ray=({ax_ray:+.3f}, {ay_ray:+.3f})")

    window_total_px = (ady2 - ady1) * (adx2 - adx1)

    def white_count(rgb_crop):
        """Count near-white (table) pixels: R, G, AND B all above
        WHITE_RGB_THRESH.  Cube body fails the B channel (cube b is much
        lower than r,g); gripper is gray (~128) and fails the per-channel
        threshold; cube shadow on table darkens R/G/B uniformly so it also
        fails."""
        return int(np.sum(np.all(rgb_crop > WHITE_RGB_THRESH, axis=-1)))

    # Capture initial white_count from the CURRENT frame at the start of
    # Phase B2 (after Phase A/B1). This keeps the baseline aligned with the
    # active attention-window content instead of using reset-time frame 0.
    phase2_start_frame = env.get_frame(obs)
    initial_crop = phase2_start_frame[ady1:ady2, adx1:adx2]
    initial_white_count = white_count(initial_crop)
    print(f"  initial white_count (Phase B2 start) = {initial_white_count} / "
          f"{window_total_px} px")

    for step_i in range(PHASE2_MAX_STEPS):
        tip_pos = env.get_gripper_tip_pos()
        # Current tip camera-frame depth (negative; camera looks along -cam_Z).
        p_cam_eef = cam_R.T @ (tip_pos - cam_pos)
        # Step "deeper" along the ray: p_cam[2] becomes more negative.
        z_cam_target = float(p_cam_eef[2]) - SERVO_DZ
        target_cam = np.array([ax_ray * z_cam_target,
                               ay_ray * z_cam_target,
                               z_cam_target])
        target_world = cam_pos + cam_R @ target_cam
        err = target_world - tip_pos
        action = np.zeros(7)
        action[:3] = np.clip(err / 0.05, -1.0, 1.0)
        action[6] = 1.0  # keep gripper closed so fingers stay at the tip site
        obs, _, _, _ = env.step(action)
        frame = record(obs, env.world_to_pixel(env.get_gripper_tip_pos()))

        # 20×20 thirdview crop at the attention window
        crop_rgb = frame[ady1:ady2, adx1:adx2]

        if step_i % 25 == 0:
            tp = env.get_gripper_tip_pos()
            tip_px_now = env.world_to_pixel(tp)
            print(f"  step={step_i:3d}  tip=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:.3f})  "
                  f"tip px={tip_px_now}  front-face px={sam3_bottom_px}")

        if step_i >= PHASE2_WARMUP_STEPS:
            wc = white_count(crop_rgb)
            frac = wc / max(window_total_px, 1)
            rise = wc - initial_white_count
            if step_i % 25 == 0:
                print(f"  white_count={wc:3d}/{window_total_px}  "
                      f"frac={frac:.2f}  rise={rise:+d}  "
                      f"(trigger when frac>{WHITE_TRIGGER_FRACTION} and "
                      f"rise>{WHITE_MIN_RISE})")
            if frac > WHITE_TRIGGER_FRACTION and rise > WHITE_MIN_RISE:
                print(f"\n--- Cube exited attention window at step {step_i}, "
                      f"tip z={env.get_gripper_tip_pos()[2]:.4f} ---")
                print(f"  white_count: {initial_white_count} → {wc}  "
                      f"(frac {frac:.2f}, rise {rise:+d})")
                # SAM3 re-segmentation on the current thirdview frame, so
                # we can show the user that the cube's mask centroid has
                # shifted from its original position (the "compare SAM3
                # segmentations from different frames" check).
                print("Running SAM3 on current thirdview frame for comparison…")
                _, verify_centroid_px, _ = segment_cube_sam3(frame, processor)
                shift = float(np.hypot(
                    verify_centroid_px[0] - centroid_px[0],
                    verify_centroid_px[1] - centroid_px[1],
                ))
                print(f"  original SAM3 centroid: {centroid_px}")
                print(f"  current  SAM3 centroid: {verify_centroid_px}")
                print(f"  pixel shift           : {shift:.1f} px")
                print("====== CUBE MOVEMENT CONFIRMED ======")
                cube_now = env.get_cube_pos()
                print(f"\nCube 3D position (simulator ground truth):")
                print(f"  x = {cube_now[0]:+.4f} m")
                print(f"  y = {cube_now[1]:+.4f} m")
                print(f"  z = {cube_now[2]:+.4f} m")
                print(f"  delta from start: {(cube_now - actual_cube).round(4)}")
                contact_confirmed = True
                break

    if not contact_confirmed:
        print(f"[Phase B2] Max iterations reached.  tip z={env.get_gripper_tip_pos()[2]:.4f}")
        cube_now = env.get_cube_pos()
        print(f"  cube position at end : {cube_now.round(4)}")
        print(f"  cube delta from start: {(cube_now - actual_cube).round(4)}")

    print(f"\nTotal frames: {len(thirdview_frames)}")
    return thirdview_frames, attention_frames, sideview_frames


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

    thirdview_frames, attention_frames, sideview_frames = run_episode(
        env, processor, out_dir
    )
    env.close()

    print("\nSaving videos...")
    save_video(thirdview_frames, str(out_dir / "thirdview.mp4"), FPS)
    save_video(attention_frames, str(out_dir / "attention.mp4"), FPS,
               upscale=ATTN_VIDEO_UPSCALE)
    save_video(sideview_frames, str(out_dir / "sideview.mp4"), FPS)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
