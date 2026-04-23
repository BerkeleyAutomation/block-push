"""
test_contact.py — verify whether cube_contact_detection fires on real contact
                  or on an arbitrary stop.

Strategy: run Phase B to completion WITHOUT breaking early, recording per-frame
metrics (motion ratio, mean optical flow, contact flag, EEF z height). At the
end we can see whether the detection fired near the moment the gripper reached
the cube, or fired early/late/never.

Outputs:
  contact_debug.mp4  — per-frame 4-panel debug video (prev | current | diff | flow)
  contact_summary.png — annotated metric strip: ratio, mean-mag, contact events,
                        and EEF height over Phase B frames
"""

import numpy as np
import cv2
import imageio

from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z
from run_push import (
    segment_cube_canny,
    cube_contact_detection,
    step_toward,
    DETECT_HALF,
    ATTN_V_OFFSET,
)

# ── env setup ────────────────────────────────────────────────────────────────
env = FrankaCubePushEnv(
    camera_height=480, camera_width=480,
    has_offscreen_renderer=True, use_camera_obs=True,
    has_renderer=False, horizon=2000, ignore_done=True,
)
obs = env.reset()

cube_pos = env.get_cube_pos()
contact_3d = np.array([cube_pos[0], cube_pos[1], TABLE_SURFACE_Z])
u0, v0 = env.world_to_pixel(contact_3d)
v0 += ATTN_V_OFFSET

H_img, W_img = 480, 480
dx1 = max(u0 - DETECT_HALF, 0)
dy1 = max(v0 - DETECT_HALF, 0)
dx2 = min(u0 + DETECT_HALF, W_img)
dy2 = min(v0 + DETECT_HALF, H_img)

print(f"Cube pos       : {cube_pos}")
print(f"Cube top z     : {cube_pos[2] + 0.02:.4f} m  (approx)")
print(f"Phase-B target : {cube_pos[2] + 0.02:.4f} m  (touch = cube_pos + 0.02)")
print(f"Attention px   : ({u0}, {v0})")

# ── Phase A ───────────────────────────────────────────────────────────────────
print("\nPhase A: moving above cube...")
above = cube_pos + np.array([0.0, 0.0, 0.20])
for obs in step_toward(env, obs, above, gripper=1.0, max_steps=100):
    pass

# ── Phase B — full descent, no early break ───────────────────────────────────
print("Phase B: descending (no early break — recording all frames)...")
touch = cube_pos + np.array([0.0, 0.0, 0.02])

prev_detect_frames = []
contact_state = {'contact_frame_count': 0}

# per-frame logs
log_ratio      = []
log_mean_mag   = []
log_max_mag    = []
log_contact    = []   # True if contact_detected fired this frame
log_eef_z      = []
debug_frames   = []   # BGR frames for the video

frame_idx = 0
for obs in step_toward(env, obs, touch, gripper=-1.0, max_steps=120):
    frame = env.get_frame(obs)   # RGB
    eef_z = float(obs["robot0_eef_pos"][2])

    # Segment + masked grayscale crop
    binary_mask = segment_cube_canny(frame, (u0, v0))
    crop_gray = cv2.cvtColor(
        cv2.cvtColor(frame[dy1:dy2, dx1:dx2], cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2GRAY,
    )
    mask_crop = binary_mask[dy1:dy2, dx1:dx2]
    crop_gray = cv2.bitwise_and(crop_gray, crop_gray, mask=mask_crop)

    prev_detect_frames.append(crop_gray)
    if len(prev_detect_frames) > 5:
        prev_detect_frames.pop(0)

    if len(prev_detect_frames) >= 4:
        # Run detection but DON'T break — just log
        contact_confirmed, debug_vis = cube_contact_detection(
            prev_detect_frames[:-1], crop_gray, contact_state)

        # Extract last printed ratio / mean_mag from the detection internals
        # by re-deriving them (same logic, no side effects)
        prev_gray = np.median(np.array(prev_detect_frames[:-1]), axis=0).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, crop_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_threshold = 2.0
        mag_filt = mag.copy()
        mag_filt[mag_filt < flow_threshold] = 0
        ratio    = float(np.sum(mag_filt > flow_threshold) / (mag.size + 1e-5))
        mean_mag = float(np.mean(mag_filt))
        max_mag  = float(np.max(mag_filt))

        log_ratio.append(ratio)
        log_mean_mag.append(mean_mag)
        log_max_mag.append(max_mag)
        log_contact.append(contact_confirmed)
        log_eef_z.append(eef_z)

        # Annotate debug_vis with frame index and EEF z
        vis = debug_vis.copy()
        cv2.putText(vis, f"f={frame_idx:03d}  eef_z={eef_z:.4f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if contact_confirmed:
            cv2.putText(vis, "CONTACT!", (4, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        debug_frames.append(vis)

    frame_idx += 1

env.close()

# ── save debug video ──────────────────────────────────────────────────────────
if debug_frames:
    writer = imageio.get_writer("contact_debug.mp4", fps=10,
                                codec="libx264", quality=8, macro_block_size=1)
    for f in debug_frames:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"\nSaved contact_debug.mp4  ({len(debug_frames)} frames)")

# ── summary image (metric strip) ─────────────────────────────────────────────
n = len(log_ratio)
if n > 0:
    W_strip = max(n * 4, 400)
    H_strip = 200
    strip = np.ones((H_strip, W_strip, 3), dtype=np.uint8) * 30  # dark background

    def to_px_y(val, lo, hi):
        val = np.clip(val, lo, hi)
        return int(H_strip - 1 - (val - lo) / (hi - lo + 1e-9) * (H_strip - 10))

    # Draw grid lines
    for val, color in [(0.15, (60, 60, 60)), (0.4, (60, 60, 60))]:
        py = to_px_y(val, 0.0, 1.0)
        cv2.line(strip, (0, py), (W_strip, py), color, 1)

    # Plot ratio (green), mean_mag (blue), eef_z scaled (yellow)
    eef_lo = min(log_eef_z) - 0.005
    eef_hi = max(log_eef_z) + 0.005
    for i in range(1, n):
        x0 = (i - 1) * W_strip // n
        x1 = i * W_strip // n

        cv2.line(strip,
                 (x0, to_px_y(log_ratio[i-1], 0.0, 1.0)),
                 (x1, to_px_y(log_ratio[i],   0.0, 1.0)),
                 (0, 200, 0), 1)  # green = ratio

        cv2.line(strip,
                 (x0, to_px_y(log_mean_mag[i-1], 0.0, 1.0)),
                 (x1, to_px_y(log_mean_mag[i],   0.0, 1.0)),
                 (200, 100, 0), 1)  # blue = mean_mag

        cv2.line(strip,
                 (x0, to_px_y(log_eef_z[i-1], eef_lo, eef_hi)),
                 (x1, to_px_y(log_eef_z[i],   eef_lo, eef_hi)),
                 (0, 200, 200), 1)  # yellow = eef_z

    # Mark contact events
    for i, c in enumerate(log_contact):
        if c:
            x = i * W_strip // n
            cv2.line(strip, (x, 0), (x, H_strip), (0, 0, 255), 2)
            cv2.putText(strip, "CONTACT", (max(x - 25, 0), H_strip - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Legend
    cv2.putText(strip, "green=ratio  blue=mean_mag  yellow=eef_z  red=contact",
                (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.imwrite("contact_summary.png", strip)
    print("Saved contact_summary.png")

# ── text summary ──────────────────────────────────────────────────────────────
contact_frames = [i for i, c in enumerate(log_contact) if c]
cube_top_z = float(cube_pos[2]) + 0.02

print("\n======= CONTACT DETECTION SUMMARY =======")
print(f"  Phase-B frames analysed : {n}")
print(f"  Cube top z (approx)     : {cube_top_z:.4f} m")
print(f"  EEF z at frame 0        : {log_eef_z[0]:.4f} m  (start of descent)")
print(f"  EEF z at last frame     : {log_eef_z[-1]:.4f} m  (end of descent)")

if contact_frames:
    for fi in contact_frames:
        print(f"  *** CONTACT fired at frame {fi:3d}  EEF z = {log_eef_z[fi]:.4f} m"
              f"  (cube top = {cube_top_z:.4f} m,"
              f"  gap = {log_eef_z[fi] - cube_top_z:+.4f} m)")
    print("\n  Verdict: contact detection DID fire.")
    gap = log_eef_z[contact_frames[0]] - cube_top_z
    if abs(gap) < 0.015:
        print("  The first trigger was close to the cube top — looks like a real contact event.")
    else:
        print(f"  The first trigger was {gap:+.4f} m from cube top — may be a false positive.")
else:
    print("\n  Verdict: contact detection did NOT fire during Phase B.")
    print("  The robot stopped because step_toward reached its tolerance (tol=0.005 m),")
    print("  NOT because motion was detected. Consider lowering detection thresholds.")
print("==========================================")
