# block-push

Franka Panda cube-pushing simulation using [RoboSuite](https://github.com/ARISE-Initiative/robosuite).

A scripted arm pushes a yellow cube from front to back relative to a fixed third-view camera. Two streams are recorded:
- **thirdview.mp4** — 480×480 full camera view
- **attention.mp4** — fixed 20×20 pixel attention window anchored at the cube-table contact area

The attention window concept is from the [Surgical D-Knot paper](https://arxiv.org/abs/2408.00191) (Fig. 3, Section C): a small fixed crop centered on the object-surface contact point that provides a local feature signal for servoing.

## Scene layout

```
Camera (front-elevated, looking toward +Y)
        |
        v
   [Yellow cube]   ← arm approaches here
   ─ ─ ─ ─ ─ ─   ← attention window fixed here (cube-table contact)
   [  Table   ]
         (back of table, +Y)
```

## Requirements

- Python 3.12+ — required for SAM3.
- [uv](https://github.com/astral-sh/uv) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A display or virtual framebuffer for MuJoCo offscreen rendering.  
  On headless servers use: `export MUJOCO_GL=egl` or `Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1`

## Setup

  # For this workstation (as dependencies already installed): 
  ```bash 
  source activate_bigdrive.sh
  ```

```bash
git clone <repo-url>
cd block-push

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

#Install SAM3 locally 
git clone https://github.com/facebookresearch/sam3.git
echo "sam3/" >> .gitignore

uv pip install -e ./sam3
uv pip install einops pycocotools decord opencv-python matplotlib scikit-image scikit-learn

```

## Usage

### 1. Verify scene geometry (recommended first step)

```bash
uv run python visualize_setup.py
```

- Opens a Viser 3D viewer at **http://localhost:8080** showing the table, cube, camera frustum, and attention anchor point.
- Saves `setup_check.png` — the rendered camera view with the attention window rectangle drawn in red.
- Press `Ctrl+C` to exit.

### 2. Run the push and record

```bash
uv run python run_push.py
```

Outputs:
| File | Description |
|------|-------------|
| `thirdview.mp4` | 480×480 third-view camera, 20 fps |
| `attention.mp4` | 20×20 attention window, 20 fps |

Console output shows per-phase progress and final cube displacement.

### 3 . Run SAM3 inference
```bash
# run with HF-default behavior via: 
uv run python sam3_setup_test.py 
# Or use local weights when you want to: 
uv run python sam3_setup_test.py --checkpoint /path/to/sam3.pt
```

## Project structure

```
block-push/
├── pyproject.toml          # uv project dependencies
├── uv.lock                 # uv lockfile
├── cube_push_env.py        # FrankaCubePushEnv
├── visualize_setup.py      # Scene/camera/attention-window visualization
├── run_push.py             # Scripted push + video recording
├── sam3_setup_test.py      # SAM3 segmentation test on setup_check.png
├── sam3/                   # Local SAM3 clone, gitignored
├── activate_bigdrive.sh    # Optional workstation cache/env activation
└── README.md
```

## Key parameters

| Parameter | Value | File |
|-----------|-------|------|
| Camera position | `[0, -1.1, 1.4]` | `cube_push_env.py` |
| Camera FOV | 45° | `cube_push_env.py` |
| Cube half-extent | 0.025 m (5 cm) | `cube_push_env.py` |
| Cube friction | `(0.05, 0.005, 0.0001)` | `cube_push_env.py` |
| Cube density | 100 kg/m³ | `cube_push_env.py` |
| Attention window | 20×20 px | `run_push.py` |
| Recording FPS | 20 | `run_push.py` |

## Troubleshooting

**Videos turn black after a few frames** — this usually indicates an unstable OSMesa path.  
The project now forces `MUJOCO_GL=egl` internally for stable offscreen rendering.

**`No module named 'robosuite'`** — run `uv sync` first.

**Arm doesn't reach cube** — OSC controller may need more steps. Increase `max_steps` in the `step_toward` calls in `run_push.py`.
