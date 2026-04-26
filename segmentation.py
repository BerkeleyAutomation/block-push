"""
segmentation.py — SAM3-based cube detection.

Usage:
    model, processor = load_sam3_model()          # once at startup
    mask, centroid_px, attn_anchor_px = segment_cube_sam3(rgb_frame, processor)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from contextlib import nullcontext

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


_PROMPTS = [
    "yellow cube",
    "pale yellow cube",
    "cube",
    "block",
    "yellow block",
]


def _resolve_bpe_path() -> str:
    candidate = (
        Path(__file__).resolve().parent
        / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"SAM3 BPE file not found: {candidate}")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_sam3_model(checkpoint_path=None):
    """
    Load SAM3 and return (model, processor). Call once at startup — expensive.

    Args:
        checkpoint_path: local .pt file, or None to load from HuggingFace.
    """
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        checkpoint_path = str(p)
    else:
        print("SAM3: loading weights from HuggingFace (no local checkpoint given)")

    model = build_sam3_image_model(
        bpe_path=_resolve_bpe_path(),
        checkpoint_path=checkpoint_path,
        load_from_HF=True,
    )
    model.eval()
    return model, Sam3Processor(model)


def segment_cube_sam3(image, processor):
    """
    Detect the cube in an RGB frame using SAM3 text prompts.

    Args:
        image:     RGB uint8 numpy array (H, W, 3) or PIL Image
        processor: Sam3Processor from load_sam3_model()

    Returns:
        mask:           bool (H, W) binary cube mask
        centroid_px:    (u, v) mask centroid — robot XY target pixel
        attn_anchor_px: (u, v) bottom-edge midpoint — contact/attention anchor
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    best = None
    with torch.no_grad(), autocast_ctx:
        state = processor.set_image(image)
        for prompt in _PROMPTS:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = _to_numpy(output["masks"])
            scores = _to_numpy(output["scores"])
            if masks.ndim == 4:
                masks = masks[:, 0]
            if len(scores) == 0:
                continue
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            print(f"  SAM3 prompt={prompt!r:25s}  score={score:.4f}")
            if best is None or score > best["score"]:
                best = {"score": score, "mask": masks[idx]}

    if best is None:
        raise RuntimeError("SAM3: no cube detected with any prompt")

    mask = best["mask"] > 0.05
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("SAM3: mask is empty after threshold")

    centroid_px = (int(np.round(xs.mean())), int(np.round(ys.mean())))

    bottom_y = int(ys.max())
    top_y = int(ys.min())
    band = max(4, int(0.04 * (bottom_y - top_y + 1)))
    bottom_x = int(np.median(xs[ys >= bottom_y - band]))
    attn_anchor_px = (bottom_x, bottom_y)

    print(
        f"SAM3: best_score={best['score']:.4f}  "
        f"centroid={centroid_px}  attn_anchor={attn_anchor_px}"
    )
    return mask, centroid_px, attn_anchor_px
