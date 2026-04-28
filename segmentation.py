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
    return model, Sam3Processor(model, confidence_threshold=0.05)


def segment_cube_sam3(image, processor, crop_center=None, crop_half=80, upscale=4):
    """
    Detect the cube in an RGB frame using SAM3 text prompts.

    SAM3 struggles with small objects. When ``crop_center=(u, v)`` is given,
    a (2*crop_half+1) square is cropped around it, upscaled by ``upscale``,
    then SAM3 runs on that zoomed view. Returned pixel coordinates are
    always mapped back to the input frame's coordinate system.

    Args:
        image:       RGB uint8 numpy array (H, W, 3) or PIL Image
        processor:   Sam3Processor from load_sam3_model()
        crop_center: (u, v) pixel in input frame to center the SAM3 window on
        crop_half:   half-size of the square crop (pixels in input frame)
        upscale:     factor to upscale the crop by before SAM3 inference

    Returns:
        mask:           bool (H, W) binary cube mask in input-frame coords
        centroid_px:    (u, v) mask centroid in input-frame coords
        attn_anchor_px: (u, v) bottom-edge midpoint in input-frame coords
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    orig_w, orig_h = image.size

    if crop_center is not None:
        cu, cv = int(crop_center[0]), int(crop_center[1])
        x1 = max(0, cu - crop_half)
        y1 = max(0, cv - crop_half)
        x2 = min(orig_w, cu + crop_half + 1)
        y2 = min(orig_h, cv + crop_half + 1)
        crop = image.crop((x1, y1, x2, y2))
        cw, ch = crop.size
    else:
        x1, y1 = 0, 0
        crop = image
        cw, ch = orig_w, orig_h

    sam_input = crop.resize((cw * upscale, ch * upscale), Image.BILINEAR)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    best = None
    with torch.no_grad(), autocast_ctx:
        state = processor.set_image(sam_input)
        for prompt in _PROMPTS:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = _to_numpy(output["masks"])
            scores = _to_numpy(output["scores"])
            if masks.ndim == 4:
                masks = masks[:, 0]
            print(f"  SAM3 prompt={prompt!r:25s}  n_dets={len(scores)}  scores={scores[:5] if len(scores)>0 else '[]'}")
            if len(scores) == 0:
                continue
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            if best is None or score > best["score"]:
                best = {"score": score, "mask": masks[idx]}

    if best is None:
        raise RuntimeError("SAM3: no cube detected with any prompt")

    mask_up = best["mask"] > 0.01
    ys, xs = np.where(mask_up)
    if len(xs) == 0:
        raise RuntimeError("SAM3: mask is empty after threshold")

    # Map upscaled-crop coords back to the input frame
    centroid_px = (
        int(np.round(xs.mean() / upscale)) + x1,
        int(np.round(ys.mean() / upscale)) + y1,
    )

    bottom_y = int(ys.max())
    top_y = int(ys.min())
    band = max(4, int(0.04 * (bottom_y - top_y + 1)))
    bottom_x = int(np.median(xs[ys >= bottom_y - band]))
    attn_anchor_px = (
        int(round(bottom_x / upscale)) + x1,
        int(round(bottom_y / upscale)) + y1,
    )

    # Downsample mask to crop size, then paste into a full-frame mask
    mask_pil = Image.fromarray(mask_up.astype(np.uint8) * 255)
    mask_pil = mask_pil.resize((cw, ch), Image.NEAREST)
    crop_mask = np.array(mask_pil) > 0
    mask = np.zeros((orig_h, orig_w), dtype=bool)
    mask[y1:y1 + ch, x1:x1 + cw] = crop_mask

    print(
        f"SAM3: best_score={best['score']:.4f}  "
        f"centroid={centroid_px}  attn_anchor={attn_anchor_px}"
    )
    return mask, centroid_px, attn_anchor_px
