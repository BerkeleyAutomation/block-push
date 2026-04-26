# block-push

Closed-loop visual cube push for a Franka Panda in [robosuite](https://github.com/ARISE-Initiative/robosuite),
driven entirely by **SAM3 segmentation + 2D pixel reprojection** — no privileged 3D
cube coordinates are used in the navigation path.

The pipeline:

1. Reset the env, capture frames from a third-view camera and a fixed
   "attention" camera that views the cube from the opposite side of the arm.
2. Run SAM3 on the attention frame to get a cube mask + centroid pixel.
3. Unproject that centroid pixel onto the cube-center Z plane → cube world XY.
4. Move the EEF above the cube (Phase A), align in XY (Phase B1), descend Z
   under action saturation until contact (Phase B2), push +Y (Phase C), retract
   (Phase D).

Two camera streams are recorded per run:

- `thirdview.mp4` — the third-view camera with mask/EEF/contact-window overlays.
- `attention.mp4` — a 20×20 crop centered on the cube-table contact point,
  upscaled 20×. The attention-window concept follows the
  [Surgical D-Knot paper](https://arxiv.org/abs/2408.00191) (Fig. 3, Section C).

## Scene layout

```
                    Robot base (+Y side of table)
                         |
   Attention cam (+Y, looks back at cube) ──┐
                                            v
                                   [Yellow cube]
                                   ─ ─ ─ ─ ─ ─    ← attention window
                                   [   Table   ]
                         ^
                         |
   Thirdview cam (-Y, looks across the table)
```

## Requirements

- Python 3.12+ (required by SAM3)
- [uv](https://github.com/astral-sh/uv)
- A GPU with ≥ 8 GB free VRAM for SAM3
- An EGL-capable display or `MUJOCO_GL=egl` for offscreen rendering

## Setup

On this workstation, dependencies and HuggingFace cache are pre-configured:

```bash
source activate_bigdrive.sh
```

Fresh setup elsewhere:

```bash
git clone <repo-url>
cd block-push

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# SAM3 source clone (gitignored)
git clone https://github.com/facebookresearch/sam3.git
echo "sam3/" >> .gitignore
uv pip install -e ./sam3
uv pip install einops pycocotools decord opencv-python matplotlib scikit-image scikit-learn
```

## Running the pipeline

```bash
uv run python run_push.py [--checkpoint /path/to/sam3.pt]
```

Omit `--checkpoint` to download from HuggingFace.

Each invocation creates a new `results/run_<N>/` folder (auto-incremented from
the highest existing index) and writes:

| File | Description |
|------|-------------|
| `thirdview.mp4`         | third-view recording (480×480 @ 20 fps) with overlays |
| `attention.mp4`         | 20×20 attention crop, upscaled 20× |
| `debug_first_frame.png` | raw thirdview at episode start |
| `debug_attn_frame.png`  | raw attention frame (what SAM3 segments) |
| `debug_sam3_attn.png`   | SAM3 mask + centroid drawn on attention frame |
| `debug_sam3.png`        | projected cube keypoints on thirdview |

The console prints per-phase progress, a `[diag]` block comparing SAM3-derived
`cube_3d` against the simulator's ground truth (sanity check only — not used
for control), and the cube delta after the push.

## Project structure

```
block-push/
├── run_push.py            # main pipeline (SAM3 + reprojection + servoing)
├── cube_push_env.py       # FrankaCubePushEnv: scene, cameras, projection
├── segmentation.py        # SAM3 wrapper: load_sam3_model, segment_cube_sam3
├── servoing.py            # pixel ↔ world math: unproject + image Jacobian
├── plan.md                # original integration plan
├── README.md
├── pyproject.toml         # uv project + dependencies
├── uv.lock
├── activate_bigdrive.sh   # workstation cache/env activation
├── results/run_<N>/       # per-run outputs (videos + debug PNGs)
├── test/                  # standalone scripts and historical artifacts
│   ├── visualize_setup.py # Viser 3D scene viewer + setup_check.png renderer
│   ├── sam3_setup_test.py # SAM3 sanity check on a static frame
│   ├── test_canny.py      # legacy Canny-based detection probe
│   ├── test_contact.py    # contact-detector tuning harness
│   └── *.png, *.mp4       # historical reference outputs
└── sam3/                  # local SAM3 source clone (gitignored)
```

### Module reference

**`cube_push_env.FrankaCubePushEnv`** — robosuite ManipulationEnv with two
fixed cameras (`thirdview_cam`, `attn_cam`), a deterministically-placed yellow
cube, and OSC-pose control. Exposes:

| Method | Purpose |
|--------|---------|
| `get_frame(obs)` / `get_attention_frame(obs)` | Extract RGB frames (top-down). |
| `get_camera_params()` / `get_attn_camera_params()` | `(cam_pos, cam_R, f, W, H)` bundles for projection math. |
| `world_to_pixel(p)` / `attn_world_to_pixel(p)` | OpenGL pinhole projection — camera looks along -Z, `v_gl = -f·Y/Z + H/2`, then flipped to top-down screen coords. |
| `get_cube_pos()` | **Diagnostic only.** Ground-truth cube position; never used for control. |

**`segmentation`** — minimal SAM3 wrapper.

| Function | Purpose |
|----------|---------|
| `load_sam3_model(checkpoint=None)` | Build the image model + processor (call once; expensive). |
| `segment_cube_sam3(image, processor)` | Run several text prompts, return `(mask, centroid_px, attn_anchor_px)` for the highest-scoring detection. |

**`servoing`** — projection geometry, no robosuite dependency.

| Function | Purpose |
|----------|---------|
| `pixel_to_world_at_z(u, v, z, cam_pos, cam_R, f, W, H)` | Exact inverse of `world_to_pixel`: shoots a ray through the pixel and intersects the world `z = z_world` plane. |
| `pixel_error_to_robot_delta(e_u, e_v, cam_pos, cam_R, eef_pos, f, Kp)` | Solve the 2×2 image Jacobian for an XY world delta that drives a pixel error to zero. Currently unused (Phase B1 navigates in world XY directly), kept for re-enabling pixel-space servoing. |

**`run_push.py`** — orchestrates the episode and recording. Tunable
parameters live near the top of the file:

| Parameter | Meaning |
|-----------|---------|
| `APPROACH_HEIGHT`     | Phase A target Z above the cube top (m). |
| `DESCENT_ACTION_MAX`  | Cap on `action[2]` magnitude in Phase B2. |
| `DESCENT_DEEP_Z`      | Phase B2 commanded Z target (well below the table — drives action saturation). |
| `STALL_WINDOW`        | Iterations of Z history compared for stall-based contact detection. |
| `STALL_DZ_THRESH`     | Minimum Z drop in `STALL_WINDOW` steps before a stall is called. |
| `ATTENTION_HALF`      | Half-side of the recorded attention crop (px). |
| `DETECT_HALF`         | Half-side of the contact-detection crop (px). |
| `FPS`                 | Output video frame rate. |

## Key scene constants (cube_push_env.py)

| Parameter | Value |
|-----------|-------|
| Thirdview camera | pos `[0, -1.1, 1.4]`, fovy 45° |
| Attention camera | pos `[0,  0.40, 0.90]`, fovy 20° |
| Cube half-extent | 0.025 m |
| Cube friction    | `(0.05, 0.005, 0.0001)` |
| Cube density     | 100 kg/m³ |
| Initial cube pos | `(0, -0.10, 0.825)` |

## Troubleshooting

**`CUDA out of memory`** — another process (e.g. a Ray Serve deployment) is
holding the GPU. Identify owners via
`nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader`
and free what you control.

**Black frames in recordings** — usually an OSMesa fallback. The env forces
`MUJOCO_GL=egl` at import time; verify the variable is `egl` in your shell.

**SAM3 detects something other than the cube** — check
`results/run_<N>/debug_sam3_attn.png`. The attention camera was added precisely
because the thirdview is occasionally fooled by the robot arm; if SAM3 still
misfires, adjust the prompt list in `segmentation._PROMPTS`.

**Phase B2 reports max iterations without contact** — the EEF didn't descend
far enough. Increase `DESCENT_ACTION_MAX` or check that Phase A leaves the EEF
properly above the cube (`above_3d` print).
