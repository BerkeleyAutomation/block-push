"""
test_canny.py — visual sanity-check for segment_cube_canny.

Saves:
  canny_check.png  — 4-panel: full frame | detection crop | binary mask | overlay
"""

import numpy as np
import cv2
from cube_push_env import FrankaCubePushEnv, TABLE_SURFACE_Z
from run_push import segment_cube_canny, DETECT_HALF, ATTN_V_OFFSET

env = FrankaCubePushEnv(
    camera_height=480, camera_width=480,
    has_offscreen_renderer=True, use_camera_obs=True,
    has_renderer=False, horizon=2000, ignore_done=True,
)
obs = env.reset()
frame = env.get_frame(obs)  # (480, 480, 3) RGB

cube_pos = env.get_cube_pos()
contact_3d = np.array([cube_pos[0], cube_pos[1], TABLE_SURFACE_Z])
u0, v0 = env.world_to_pixel(contact_3d)
v0 += ATTN_V_OFFSET
env.close()

print(f"Attention centre: ({u0}, {v0})")

# --- run segmentation ---
binary_mask = segment_cube_canny(frame, (u0, v0))

# --- build panels ---
H, W = frame.shape[:2]
x1 = max(u0 - DETECT_HALF, 0)
y1 = max(v0 - DETECT_HALF, 0)
x2 = min(u0 + DETECT_HALF, W)
y2 = min(v0 + DETECT_HALF, H)

frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Panel 1: full frame with attention rectangle
full_ann = frame_bgr.copy()
cv2.rectangle(full_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.circle(full_ann, (u0, v0), 4, (0, 0, 255), -1)

# Panel 2: raw detection crop
crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
crop_bgr = cv2.resize(crop_bgr, (480, 480), interpolation=cv2.INTER_NEAREST)

# Panel 3: binary mask crop (cube=white, table=black)
mask_crop = binary_mask[y1:y2, x1:x2]
mask_bgr = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
mask_bgr = cv2.resize(mask_bgr, (480, 480), interpolation=cv2.INTER_NEAREST)

# Panel 4: green overlay on full frame
overlay = frame_bgr.copy()
overlay[binary_mask > 0] = (0, 200, 0)
blended = cv2.addWeighted(frame_bgr, 0.6, overlay, 0.4, 0)
cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Add labels
def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)

label(full_ann,  "1. full frame + window")
label(crop_bgr,  "2. detection crop (raw)")
label(mask_bgr,  "3. binary mask (cube=white)")
label(blended,   "4. green overlay")

out = np.hstack([full_ann, crop_bgr, mask_bgr, blended])
cv2.imwrite("canny_check.png", out)
print("Saved canny_check.png")

white_px = int(np.sum(binary_mask > 0))
print(f"White pixels in mask: {white_px}  (0 = cube not found in window)")
