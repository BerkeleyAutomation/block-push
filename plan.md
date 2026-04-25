# Integration Plan: SAM3 + Visual Servoing Pipeline

**Status as of 2026-04-25**
- SAM3 segmentation working (`sam3_setup_test.py`)
- Initial block push working (`run_push.py`) with Canny-based detection and hardcoded motion
- Goal: replace Canny with SAM3 outputs and replace hardcoded motion with closed-loop visual servoing

---

## Step 1 — Wire SAM3 into the Pipeline

**What:** Extract the SAM3 logic from `sam3_setup_test.py` into a reusable function and call it from `run_push.py` instead of `segment_cube_canny`.

**Inputs / Outputs:**
- Input: RGB camera frame (480×480 numpy array)
- Outputs:
  - `cube_mask`: binary H×W array
  - `centroid_px`: `(u, v)` — robot tool target (currently "probe_point" in sam3_setup_test)
  - `attn_anchor_px`: `(u, v)` — bottom-edge midpoint, used as attention window center

**Tasks:**
1. Create `def segment_cube_sam3(image, processor, state) -> (mask, centroid_px, attn_anchor_px)` in a shared module (e.g., add to `run_push.py` or new `segmentation.py`).
2. Initialize the SAM3 model once at episode start (expensive; do not re-init per frame).
3. Drop `segment_cube_canny` from the live path (keep for comparison / fallback if needed).
4. Validate: overlay mask + keypoints on a captured frame and confirm they match the cube.

**Reference:** `sam3_setup_test.py` lines that compute `bottom_x`, `probe_point`, `attention_window_center`.

---

## Step 2 — Dynamic Attention Window

**What:** Replace the fixed `ATTN_V_OFFSET`-based attention anchor with the SAM3 bottom-edge midpoint.

**Tasks:**
1. At the start of each episode, run SAM3 on the first frame to get `attn_anchor_px`.
2. Pass `attn_anchor_px` into `cube_contact_detection` as the crop center (replaces hardcoded `attn_px`).
3. Re-enable / verify the attention window rectangle drawing in visualization code (currently drawn in the servoing loop; confirm it renders in `run_push.py`'s debug frames).
4. Optionally update every N frames if the cube is expected to shift before contact.

**Stopping condition clarification:**
- Stop on contact (motion in attention window) only.
- Remove any leftover "cube left frame" logic — it is no longer needed.

---

## Step 3 — Visual Servoing (Phase B Replacement)

**What:** Replace the hardcoded `step_toward(target_pos)` descent in Phase B with a closed-loop pixel-error servo loop.

**Design (adapted from surgical servoing reference code):**

```
Phase 1 (XY alignment):
  target pixel  ← SAM3 centroid_px  (or slightly above: centroid_px - (0, OFFSET))
  while pixel_error > stop_pixel_threshold:
      project current EEF → pixel (u_now, v_now)
      e_u = target_u - u_now
      e_v = target_v - v_now
      Δrobot = pixel_error_to_robot_delta(e_u, e_v, K, T_cam_to_robot, z_robot)
      send OSC action(Δrobot)
      check contact → break if detected

Phase 2 (Z approach):
  while not contact_detected:
      step small Δz toward cube (servoing_distance ≈ 0.001 m)
      keep XY on target pixel ray (project pixel back at new Z)
      run contact detection on attention window
      stop immediately on contact
```

**Tasks:**
1. Expose camera intrinsics `K` from `FrankaCubePushEnv` (focal length from `world_to_pixel`; build 3×3 K matrix).
2. Expose `T_cam_to_robot` transform (camera pose already in env; invert for cam→world, compose with robot base).
3. Implement `pixel_error_to_robot_delta(e_u, e_v, K, T_cam_to_robot, z_robot)` — adapt `solve_robot_xy_for_pixel` from the surgical servoing code.
4. Implement Phase 1 loop with `stop_pixel_threshold = 20 px`.
5. Implement Phase 2 loop with `servoing_distance = 0.001 m` and contact break.
6. Wrap both phases in `run_episode` where Phase B currently lives.

**Key parameters to tune:**
| Parameter | Starting value | Notes |
|-----------|---------------|-------|
| `Kp` | 1.0 | Proportional gain; reduce if oscillating |
| `stop_pixel_threshold` | 20 px | XY done when within this |
| `servoing_distance` | 0.001 m | Z step per iteration |
| `position_correction` | 0.002 m | Z offset at contact to avoid overshoot |

---

## Step 4 — Contact Detection Validation & Tuning

**What:** Confirm that the attention-window contact detector reliably fires on first contact and does not false-trigger.

**Tasks:**
1. Run `test_contact.py`-style recording with the new SAM3 attention anchor and check `contact_summary.png` metrics.
2. If premature triggers: raise `ratio` threshold (currently 0.15) or `mean_mag` threshold (0.4).
3. If missing contact: lower thresholds or tighten window size.
4. Confirm robot halts within one servo iteration of trigger (no coasting).

---

## Step 5 — End-to-End Pipeline Assembly

**Final episode flow:**

```
1. Reset env, capture first frame
2. SAM3 → cube_mask, centroid_px, attn_anchor_px
3. Phase A: move EEF to approach height above centroid_px
4. Phase B (servo):
     Phase B1: XY align to centroid_px (pixel servo)
     Phase B2: Z approach until contact at attn_anchor_px
5. Phase C: small push (+Y, fixed distance ~5 cm)
6. Phase D: retract
7. (hook) → grasp phase
```

**Tasks:**
1. Stitch all phases together in `run_episode`.
2. Preserve dual-camera recording (thirdview + attention window).
3. Add per-frame overlay: SAM3 mask, centroid dot, attention window rectangle, EEF pixel projection.
4. Run a full episode and review the recorded video.

---

## Step 6 — Post-Contact / Grasp Hook (Future Phase Prep)

**What:** After the stop in Phase B2, set up the robot pose for grasp execution.

**Tasks:**
1. Log EEF pose at contact stop.
2. Add a small upward correction (`position_correction_for_hitting_thread ≈ 0.002 m`) to seat the gripper.
3. Add placeholder `grasp_phase()` function that opens/closes gripper — leave as stub until grasp policy is ready.
4. Verify EEF orientation at contact is valid for grasping (quaternion check).

---

## Environment / Infra Checklist

- [ ] Activate `activate_bigdrive.sh` before running SAM3 (GPU memory)
- [ ] SAM3 checkpoint path configured (local or HuggingFace fallback)
- [ ] `sam3_setup_test.py` passes with expected mask quality on current scene
- [ ] `pyproject.toml` dependencies include opencv-python, imageio[ffmpeg]
- [ ] All new functions have minimal inline validation (print centroid, assert mask non-empty)

---

## File Change Summary

| File | Changes |
|------|---------|
| `run_push.py` | Add SAM3 init, replace Canny call, add servo loop (Phase B), update `run_episode` |
| `cube_push_env.py` | Expose `K` matrix and `T_cam_to_robot` as properties |
| `sam3_setup_test.py` | Refactor detection logic into importable `segment_cube_sam3()` |
| *(new)* `servoing.py` | `pixel_error_to_robot_delta`, `solve_robot_xy_for_pixel` (ported from surgical code) |

---

## Order of Work

1. **Step 1** — SAM3 function refactor (unblocks everything else)
2. **Step 3a** — Expose K and T_cam_to_robot from env (needed for servoing math)
3. **Step 2** — Dynamic attention window (can test contact detection independently)
4. **Step 3b** — Phase 1 XY servo loop
5. **Step 4** — Contact detection tuning with new anchor
6. **Step 3c** — Phase 2 Z servo loop
7. **Step 5** — Full pipeline assembly and video review
8. **Step 6** — Grasp hook stub
