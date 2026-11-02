import os
import numpy as np
from PIL import Image
import torch

_clip_model = None
_clip_processor = None

def _get_clip_validator():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


PROMPTS = [
    "a medical frontal chest x-ray radiograph showing human lungs and rib cage",
    "a document, signature, invoice, handwriting, photo of a person, animal, everyday object, screenshot, drawing, or non-chest image"
]


def validate_chest_xray(image: Image.Image) -> tuple[bool, str, dict]:
    """
    Two-stage robust Chest X-ray verification:
    Stage 1: Fast statistical & visual heuristics (color saturation, document/signature white background, contrast).
    Stage 2: Zero-shot deep neural network classifier (CLIP) to confirm the presence of a chest radiograph.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    aspect_ratio = float(w / h)
    color_std = float(np.std(arr, axis=2).mean())
    gray = arr.mean(axis=2)
    img_mean = float(gray.mean())
    img_std = float(gray.std())
    
    # 4 corners
    ch, cw = max(2, int(h * 0.05)), max(2, int(w * 0.05))
    corners = [
        float(gray[:ch, :cw].mean()),
        float(gray[:ch, -cw:].mean()),
        float(gray[-ch:, :cw].mean()),
        float(gray[-ch:, -cw:].mean())
    ]
    corner_mean = float(np.mean(corners))
    white_pct = float((gray > 220).mean() * 100.0)
    dark_pct = float((gray < 35).mean() * 100.0)
    
    metrics = {
        "aspect_ratio": round(aspect_ratio, 2),
        "chromatic_variation": round(color_std, 1),
        "mean_brightness": round(img_mean, 1),
        "contrast_std": round(img_std, 1),
        "corner_brightness": round(corner_mean, 1),
        "white_pixel_percentage": round(white_pct, 1),
    }

    # Stage 1: Fast rules
    if aspect_ratio < 0.35 or aspect_ratio > 3.0:
        return (
            False,
            f"Unusual image proportions (aspect ratio {aspect_ratio:.2f}). Chest radiographs typically have standard portrait or near-square dimensions.",
            metrics
        )

    # Document / Signature / White paper check
    if white_pct > 30.0 or corner_mean > 165.0:
        return (
            False,
            f"The image appears to be a signature, document, invoice, or drawing ({white_pct:.1f}% white background, bright borders). Real chest X-rays have dark borders.",
            metrics
        )

    # Blank / Pitch black check
    if img_std < 20.0 or (img_mean < 30.0 and dark_pct > 80.0):
        return (
            False,
            "Image is blank, flat, or nearly pitch black with no readable anatomical features.",
            metrics
        )

    # Stage 2: Deep Learning Zero-Shot Vision-Language Verification (CLIP)
    try:
        clip_model, clip_processor = _get_clip_validator()
        inputs = clip_processor(text=PROMPTS, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            cxr_confidence = float(probs[0].item() * 100.0)
            other_confidence = float(probs[1].item() * 100.0)

        metrics["chest_xray_confidence"] = f"{cxr_confidence:.1f}%"
        metrics["non_xray_confidence"] = f"{other_confidence:.1f}%"

        # Threshold: must be identified as chest x-ray with at least 65% confidence
        if cxr_confidence < 65.0:
            return (
                False,
                f"The image was identified as not being a chest X-ray ({other_confidence:.1f}% non-radiograph confidence). Please upload a valid chest X-ray.",
                metrics
            )
    except Exception as e:
        # Fallback to color check if CLIP fails
        if color_std > 12.0:
            return (
                False,
                f"Color detected ({color_std:.1f}). Chest X-rays are monochrome radiographs.",
                metrics
            )

    return True, "Valid Chest X-ray verified.", metrics
