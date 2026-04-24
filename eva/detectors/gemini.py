"""
Gemini-based object detector.

Adapted from tiptop/perception/gemini.py (https://github.com/tiptop-robot/tiptop).

Sends a PIL image + task instruction to Gemini and parses the structured JSON
response into bounding boxes and (optionally) grounded logical atoms.

Output format shared with all eva detectors:
  bboxes        — [{"box_2d": [ymin, xmin, ymax, xmax], "label": str}, ...]
                  coordinates are integers in [0, 1000]
  grounded_atoms — [{"predicate": str, "args": [str, ...]}, ...]
                   empty for the "detect" prompt; populated by prompts that
                   return predicate-level annotations (e.g. spatial relations)

Prompt templates live in eva/detectors/gemini_detect/<name>.md and are
loaded once and cached. Credentials are picked up automatically from the
GOOGLE_API_KEY environment variable via the google-genai SDK.
"""

import json
import logging
from functools import cache
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "gemini_detect"


@cache
def load_prompt(prompt_name: str) -> str:
    """Return the prompt template for `prompt_name`, loaded once from disk."""
    return (_PROMPTS_DIR / f"{prompt_name}.md").read_text().strip()


@cache
def gemini_client() -> genai.Client:
    """Return a shared Gemini client (constructed once, reused across calls)."""
    return genai.Client()


def load_json(response_text: str) -> list | dict:
    """Parse JSON from `response_text`, stripping Markdown code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    try:
        results = json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise
    return results


def _parse_response(response_text: str) -> tuple[list, list]:
    """Parse a raw Gemini response into (bboxes, grounded_atoms).

    Expects the model to return JSON with optional "bboxes" and "predicates" keys.
    Raises ValueError if the response is not valid JSON.
    """
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(
            f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}"
        )
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def detect(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-2.5-flash",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Detect task-relevant objects in `image` using Gemini.

    Args:
        image: Scene to analyse.
        task_instruction: Natural language description of the current robot task
            (e.g. "pick up the red cup and place it on the plate"). Used to
            focus detection on task-relevant objects.
        client: Gemini API client. Defaults to the shared singleton.
        model_id: Gemini model to call.
        temperature: Sampling temperature. None lets the API use its default
            (greedy for flash models). Higher values increase label diversity
            at the cost of consistency.

    Returns:
        (bboxes, grounded_atoms) where:
        - bboxes: [{"box_2d": [ymin, xmin, ymax, xmax], "label": str}, ...]
          Up to 10 objects, ordered by task relevance. Coordinates are
          integers in [0, 1000] (normalised image space).
        - grounded_atoms: always [] — the "detect" prompt returns bboxes only.
    """
    client = client or gemini_client()
    prompt = load_prompt("detect").format(task_instruction=task_instruction)
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)


async def detect_async(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-2.5-flash",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Async version of detect(); uses the Gemini async client under the hood.

    Signature and return value are identical to detect(). Prefer this inside
    async control loops to avoid blocking the event loop during network I/O.
    """
    client = client or gemini_client()
    prompt = load_prompt("detect").format(task_instruction=task_instruction)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)