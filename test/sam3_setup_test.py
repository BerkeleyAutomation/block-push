import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse
from contextlib import nullcontext

# SAM3 utilities:
# - build_sam3_image_model() creates the segmentation model
# - Sam3Processor() provides the interface for setting an image and text prompt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_numpy(x):
    """
    Convert tensors or array-like outputs into a NumPy array.

    SAM3 may return PyTorch tensors. For downstream processing
    (mask operations, indexing, visualization), NumPy is easier to use.
    """
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_bpe_path() -> str:
    """
    Resolve the SAM3 tokenizer BPE file path for local clone installs.

    In this workspace SAM3 is vendored under `./sam3`, and pkg_resources can
    fail to resolve assets depending on how the package is imported.
    """
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Could not find SAM3 BPE file at expected path: {candidate}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation on setup_check.png")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local SAM3 checkpoint (.pt). If omitted, weights are loaded from Hugging Face.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Load the rendered setup image from your robosuite scene.
    # This is the image on which SAM3 will detect the cube.
    image_path = script_dir / "setup_check.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Loading SAM3 model...")

    # Build the SAM3 image model and switch to eval mode since
    # we are only doing inference, not training.
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        print("Using Hugging Face weights (default).")

    try:
        model = build_sam3_image_model(
            bpe_path=resolve_bpe_path(),
            checkpoint_path=checkpoint_path,
            load_from_HF=True,
        )
    except Exception as e:
        msg = str(e)
        if "GatedRepoError" in msg or "gated repo" in msg.lower() or "401" in msg:
            raise RuntimeError(
                "SAM3 checkpoint download is gated on Hugging Face.\n"
                "Use one of:\n"
                "  1) Provide a local checkpoint: --checkpoint /path/to/sam3.pt\n"
                "  2) Login to HF first: huggingface-cli login\n"
                "Then rerun this script."
            ) from e
        raise
    model.eval()

    # Create a SAM3 processor, which handles the image state
    # and text-prompt-based segmentation queries.
    processor = Sam3Processor(model)

    # Try multiple prompt phrasings in case one works better than another.
    # Since your cube is yellow, these prompts are targeted to that.
    prompts = [
        "yellow cube",
        "pale yellow cube",
        "cube",
        "block",
        "block side face",
        "Pale yellow and dark yellow cubes",
        "yellow block",
    ]

    # This will store the best detection across all prompts.
    best = None

    print("Running SAM3...")

    # Disable gradients and use AMP on CUDA, which SAM3 expects for many paths.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        # Set the image once. SAM3 will internally build image features/state.
        state = processor.set_image(image)

        # Query SAM3 using each text prompt and keep the highest-scoring result.
        for prompt in prompts:
            # Ask SAM3 to segment the object described by the prompt.
            output = processor.set_text_prompt(state=state, prompt=prompt)

            # Convert outputs to NumPy for easier processing.
            masks = to_numpy(output["masks"])
            boxes = to_numpy(output["boxes"])
            scores = to_numpy(output["scores"])

            # Some SAM3 outputs may have shape [N, 1, H, W];
            # if so, squeeze out the singleton dimension.
            if masks.ndim == 4:
                masks = masks[:, 0]

            # If this prompt returned no detections, skip it.
            if len(scores) == 0:
                print(f"No detections for prompt: {prompt}")
                continue

            # Pick the highest-scoring detection for this prompt.
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            print(f"prompt={prompt}, best_score={score:.4f}")

            # If this detection is the best seen so far, keep it.
            if best is None or score > best["score"]:
                best = {
                    "prompt": prompt,
                    "score": score,
                    "mask": masks[idx],
                    "box": boxes[idx],
                }

    # If nothing was found at all, stop with an error.
    if best is None:
        raise RuntimeError("SAM3 found no cube.")

    # Convert the soft/probabilistic mask into a binary mask.
    # Lower threshold includes more uncertain cube pixels.
    mask = best["mask"] > 0.05

    # Get coordinates of all pixels belonging to the cube mask.
    ys, xs = np.where(mask)

    # If mask is empty after thresholding, fail.
    if len(xs) == 0:
        raise RuntimeError("SAM3 returned empty mask.")

    # Approximate the cube-table contact region:
    # - bottom_y = lowest row in the cube mask
    # - bottom_band = a thin band near the bottom of the mask
    # - bottom_x = median x-position in that band
    #
    # This gives a point near the center of the cube,
    # which is a good proxy for the contact area with the table.
    top_y = int(ys.min())
    bottom_y = int(ys.max())
    mask_h = bottom_y - top_y + 1
    bottom_band_thickness = max(4, int(0.04 * mask_h))
    bottom_band = ys >= bottom_y - bottom_band_thickness
    bottom_x = int(np.median(xs[bottom_band]))

    # Geometric center of the segmented cube (binary mask centroid).
    probe_x = int(np.round(float(xs.mean())))
    probe_y = int(np.round(float(ys.mean())))
    probe_point = (probe_x, probe_y)

    # Attention-window anchor: bottom edge of the mask, horizontal middle.
    # bottom_y is the lowest mask row (image y grows downward).
    # bottom_x is the median x among pixels in a thin band at that edge.
    attention_window_center = (bottom_x, bottom_y)

    # Attention window uses the same definition as above: bottom_y = lowest masked row,
    # bottom_x = median x in the bottom band (not an extra assert—logic only).
    x1, y1, x2, y2 = map(int, best["box"])
    bbox_bottom_mid_x = (x1 + x2) // 2
    print(
        "Attention-window check: mask bottom row y =",
        bottom_y,
        "bbox bottom y2 =",
        y2,
        "(delta y",
        bottom_y - y2,
        "px) | bottom_x vs bbox bottom mid x:",
        bottom_x,
        "vs",
        bbox_bottom_mid_x,
        "(delta x",
        bottom_x - bbox_bottom_mid_x,
        "px)",
    )

    # Print detection summary.
    print("Best prompt:", best["prompt"])
    print("Score:", best["score"])
    print("Box:", best["box"])
    print("Robot probe point (mask centroid):", probe_point)
    print("Attention window center (bottom-edge middle):", attention_window_center)

    # Convert PIL image to NumPy for visualization.
    img = np.array(image)
    vis = img.copy()

    # Overlay the mask in red for visualization.
    # This blends the original image with red wherever the mask is true.
    vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    # Draw the bounding box in green.
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Blue (RGB): attention window center = bottom edge of mask, horizontal middle.
    cv2.circle(vis, attention_window_center, 1, (0, 0, 255), -1)
    # Red (RGB): robot probe = mask centroid.
    cv2.circle(vis, probe_point, 1, (255, 0, 0), -1)

    # Save visualization.
    # OpenCV expects BGR ordering when writing, so convert from RGB to BGR.
    output_path = script_dir/"sam3_setup_check.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved {output_path}")

    return {
        "probe_point": probe_point,
        "attention_window_center": attention_window_center,
    }


if __name__ == "__main__":
    main()

"""
    # Also store a separate mask from a side-specific prompt.
    side_autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.no_grad(), side_autocast_ctx:
        side_output = processor.set_text_prompt(state=state, prompt="custom prompt")
    side_masks = to_numpy(side_output["masks"])
    side_scores = to_numpy(side_output["scores"])
    if side_masks.ndim == 4:
        side_masks = side_masks[:, 0]
    if len(side_scores) > 0:
        side_idx = int(np.argmax(side_scores))
        side_mask = side_masks[side_idx] > 0.1
        side_vis = img.copy()
        side_vis[side_mask] = (
            0.5 * side_vis[side_mask] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)
        side_mask_path = script_dir / "sam3_setup_check_side_mask.png"
        cv2.imwrite(str(side_mask_path), cv2.cvtColor(side_vis, cv2.COLOR_RGB2BGR))
        print(f"Saved {side_mask_path}")
    else:
        print("No detections for prompt: yellow block side face")
"""