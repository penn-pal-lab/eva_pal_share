"""
Open-source object detector using the official Grounding DINO library (IDEA-Research).

Install:
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    (requires CUDA_HOME set; falls back to CPU if CUDA unavailable)

Weights are downloaded automatically to ~/.cache/groundingdino/ on first use.
Override download directory with env var: GDINO_WEIGHTS_DIR=/your/path

Matches the gemini.py detector API:
  detect(image, task_instruction, ...) -> (bboxes, grounded_atoms)
  bboxes: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "...", "score": float}, ...]
          coordinates are integers in [0, 1000]
  grounded_atoms: always []
"""

import asyncio
import logging
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_log = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

_CONFIGS = {
    "swint": "GroundingDINO_SwinT_OGC.py",
    "swinb": "GroundingDINO_SwinB_cfg.py",
}
_WEIGHT_URLS = {
    "swint": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download"
        "/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ),
    "swinb": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download"
        "/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    ),
}
DEFAULT_BACKBONE = "swint"

_CACHE_DIR = Path(
    os.environ.get("GDINO_WEIGHTS_DIR", Path.home() / ".cache" / "groundingdino")
)


# ── Config & weight resolution ────────────────────────────────────────────────

def _config_path(backbone: str) -> str:
    import groundingdino
    cfg = Path(groundingdino.__file__).parent / "config" / _CONFIGS[backbone]
    if not cfg.exists():
        raise FileNotFoundError(
            f"Grounding DINO config not found at {cfg}. "
            "Is groundingdino installed? "
            "Run: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
        )
    return str(cfg)


def _weights_path(backbone: str) -> str:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = _WEIGHT_URLS[backbone].rsplit("/", 1)[-1]
    dest = _CACHE_DIR / fname
    if not dest.exists():
        _log.info("Downloading Grounding DINO (%s) weights to %s …", backbone, dest)
        urllib.request.urlretrieve(_WEIGHT_URLS[backbone], dest)
        _log.info("Download complete.")
    return str(dest)


# ── Model loading (cached per backbone + device) ──────────────────────────────

@lru_cache(maxsize=2)
def _load_model(backbone: str, device: str):
    from groundingdino.util.inference import load_model
    _log.info("Loading Grounding DINO (%s) onto %s …", backbone, device)
    model = load_model(_config_path(backbone), _weights_path(backbone))
    return model.to(device).eval()


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Image preprocessing ───────────────────────────────────────────────────────

def _pil_to_gdino_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Replicate groundingdino's load_image transform on a PIL image directly."""
    rgb = pil_image.convert("RGB")
    w, h = rgb.size

    # Resize: longest side → 800, hard cap at 1333
    scale = 800 / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    if max(new_w, new_h) > 1333:
        scale2 = 1333 / max(new_w, new_h)
        new_w, new_h = round(new_w * scale2), round(new_h * scale2)

    resized = rgb.resize((new_w, new_h), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor


# ── Text prompt formatting ────────────────────────────────────────────────────

def _format_prompt(text: str) -> str:
    """Normalise to Grounding DINO's period-separated format: 'cup . plate . apple .'"""
    parts = [p.strip().rstrip(".") for p in text.split(".") if p.strip()]
    return " . ".join(parts) + " ."


# ── Core inference ────────────────────────────────────────────────────────────

def _run_inference(
    image: Image.Image,
    gdino_prompt: str,
    backbone: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[list[dict], list[dict]]:
    from groundingdino.util.inference import predict
    from torchvision.ops import box_convert

    model = _load_model(backbone, device)
    image_tensor = _pil_to_gdino_tensor(image)

    # predict() returns boxes in CXCYWH, normalised 0–1 (relative to resized image,
    # but since resize is uniform the normalised coords are identical to the original)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=gdino_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # CXCYWH 0-1  →  XYXY 0-1
    if len(boxes) == 0:
        return [], []

    xyxy = box_convert(boxes.cpu(), "cxcywh", "xyxy").tolist()
    scores = logits.cpu().tolist()

    bboxes = []
    for (xmin, ymin, xmax, ymax), score, phrase in zip(xyxy, scores, phrases):
        bboxes.append({
            "box_2d": [
                round(ymin * 1000),
                round(xmin * 1000),
                round(ymax * 1000),
                round(xmax * 1000),
            ],
            "label": phrase,
            "score": round(score, 4),
        })

    return bboxes, []


# ── Public API (mirrors gemini.py) ────────────────────────────────────────────

def detect(
    image: Image.Image,
    task_instruction: str,
    text_prompt: str | None = None,
    backbone: str = DEFAULT_BACKBONE,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Detect objects in an image using Grounding DINO (local, open-source).

    Args:
        image: PIL image to analyse.
        task_instruction: Natural language task description. Used as the text
            prompt when `text_prompt` is None.
        text_prompt: Explicit detection targets, period-separated, e.g.
            "red cup . plate . banana". Overrides `task_instruction` when
            provided. Grounding DINO works best with short noun phrases rather
            than full sentences, so prefer this when you know what to detect.
        backbone: "swint" (fast, default) or "swinb" (accurate).
        box_threshold: Minimum objectness score to keep a detection (0–1).
        text_threshold: Minimum label confidence to keep a detection (0–1).
        device: "cuda", "cpu", or None to auto-detect.

    Returns:
        (bboxes, grounded_atoms) matching the gemini.py interface:
        - bboxes: [{"box_2d": [ymin, xmin, ymax, xmax], "label": str,
                    "score": float}, ...]
          Coordinates are integers in [0, 1000].
        - grounded_atoms: always [] (Grounding DINO does not output predicates).
    """
    gdino_prompt = _format_prompt(text_prompt or task_instruction)
    return _run_inference(
        image, gdino_prompt, backbone, box_threshold, text_threshold,
        device or _default_device(),
    )


async def detect_async(
    image: Image.Image,
    task_instruction: str,
    text_prompt: str | None = None,
    backbone: str = DEFAULT_BACKBONE,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Async wrapper around detect(). Runs GPU inference in a thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: detect(
            image, task_instruction, text_prompt, backbone,
            box_threshold, text_threshold, device,
        ),
    )


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import cv2

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    prompt   = sys.argv[2] if len(sys.argv) > 2 else "object"

    if img_path is None:
        print("Usage: python grounding_dino.py <image_path> [text_prompt]")
        sys.exit(1)

    pil_img = Image.open(img_path).convert("RGB")
    bboxes, _ = detect(pil_img, task_instruction=prompt)

    print(f"Detected {len(bboxes)} object(s):")
    for b in bboxes:
        print(f"  {b['label']:30s}  score={b['score']:.3f}  box={b['box_2d']}")

    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h_px, w_px = img_cv.shape[:2]
    for b in bboxes:
        ymin, xmin, ymax, xmax = b["box_2d"]
        x0 = int(xmin / 1000 * w_px); y0 = int(ymin / 1000 * h_px)
        x1 = int(xmax / 1000 * w_px); y1 = int(ymax / 1000 * h_px)
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(img_cv, f"{b['label']} {b['score']:.2f}",
                    (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 0), 1)

    out = str(Path(img_path).with_stem(Path(img_path).stem + "_gdino"))
    cv2.imwrite(out, img_cv)
    print(f"Saved annotated image → {out}")
