"""
RefSheetArtist — FLUX.2 Klein 4B distilled backend

This is the same pipeline wrapper that used to be RefSheetPipeline inside
consistency_controller.py, unchanged in behavior — just renamed and
formalized against the GenerationBackend interface so it can sit
side-by-side with other backends (e.g. an SDXL furry checkpoint).

NOTE: kept as a first-pass sketch, not verified end-to-end against your
installed diffusers version — same caveats as before around the exact
kwarg name for multi-reference images and step/guidance defaults.
"""

from __future__ import annotations

import random

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from generation_backend import GenerationBackend

DISTILLED_MODEL_ID = r"C:\AI\models\FLUX.2-klein-4B"
NUM_INFERENCE_STEPS = 4          # distilled model: few-step by design
MAX_REFERENCE_IMAGES = 2         # cap for 8GB VRAM + cpu offload, per design doc


class FluxKleinBackend(GenerationBackend):
    """Wraps Flux2KleinPipeline for this project's use."""

    max_reference_images = MAX_REFERENCE_IMAGES
    consistency_strategy = "reference_images"

    def __init__(self, model_id: str = DISTILLED_MODEL_ID, device: str = "cuda"):
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=True,       # Stops it from trying to connect to the internet
        )
        # Confirmed necessary on 8GB VRAM in testing.
        self.pipe.enable_model_cpu_offload()

    def compose_prompt(self, base_prompt: str, view_phrase: str, prompt_delta: str) -> str:
        # view_phrase intentionally unused — prompt_delta already includes
        # it, and each view's delta was individually hand-tuned (see
        # DEFAULT_VIEW_PLAN), not reducible to "view phrase + fixed suffix".
        return f"{base_prompt}, {prompt_delta}"

    def generate(
        self,
        prompt: str,
        reference_images: list[Image.Image] | None = None,
        negative_prompt_prepend: str | None = None,
        negative_prompt_append: str | None = None,
        seed: int | None = None,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ) -> tuple[Image.Image, int]:
        # FLUX.2 Klein doesn't have a real negative-prompt input the way
        # SDXL does — that's why consistency is handled via explicit
        # positive anchoring in the base prompt instead. Silently ignore
        # both negative_prompt_* args here so callers don't need
        # backend-specific branching.
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        if reference_images:
            # Multi-reference editing: pass prior view(s) as conditioning images.
            kwargs["image"] = reference_images

        result = self.pipe(**kwargs)
        image = result.images[0]
        return image, seed
