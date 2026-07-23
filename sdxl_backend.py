"""
RefSheetArtist — SDXL (furry checkpoint) backend

Fundamentally different generation flow from FluxKleinBackend, not just a
different model:

- Plain txt2img. No image-conditioning input at all — no IP-Adapter, no
  ControlNet, no reference images. Each of the four views (front, back,
  side, three-quarter) is an independent generation call.
- Character consistency across views comes entirely from reusing the
  SAME seed for every view (consistency_strategy = "shared_seed").
  ConsistencyController enforces this: it locks the seed chosen at the
  canonical stage and forces every subsequent view to reuse it.
- Prompt order is also different from FLUX: the view name comes FIRST,
  then the character prompt, then a fixed pose suffix — see
  compose_prompt() below. This is why prompt_delta (FLUX's per-view
  tuned text) is ignored here; SDXL builds its own fixed structure
  instead of using individually-tuned deltas.
- Negative prompt is real here (unlike FLUX) — composed from
  config/sdxl_negative_prompt.txt plus any per-view extras. See
  negative_prompt_config.py.

Skeleton only — fill in once you've picked a specific checkpoint (Pony
Diffusion derivative, etc.) and confirmed VRAM headroom / whether it
needs offload on your 8GB laptop.
"""

from __future__ import annotations
from pathlib import Path

import random

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from generation_backend import GenerationBackend
from negative_prompt_config import compose_negative_prompt, load_default_negative_prompt

SDXL_MODEL_PATH = Path("./config/sdxl_path.txt")
SDXL_MODEL_ID = SDXL_MODEL_PATH.read_text("utf-8")
NUM_INFERENCE_STEPS = 100          # 28 is SDXL default range, not few-step like the distilled FLUX model
GUIDANCE_SCALE = 4.0

# Fixed suffix appended after the character prompt on every view, per your
# spec: "(view) view, (character prompt), full body, T-pose, neutral expression"
POSE_SUFFIX = "full body, T-pose, neutral expression"


class SDXLBackend(GenerationBackend):
    """Plain txt2img SDXL. No reference/conditioning images — consistency
    comes entirely from reusing the same seed across all four views."""

    max_reference_images = 0
    consistency_strategy = "shared_seed"

    def __init__(self, model_id: str = SDXL_MODEL_ID, device: str = "cuda"):
        # Loaded fresh at construction time, so editing
        # config/sdxl_negative_prompt.txt takes effect on the next run
        # without any code change.
        self.default_negative_prompt = load_default_negative_prompt()

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_id,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    def compose_prompt(self, base_prompt: str, view_phrase: str, prompt_delta: str) -> str:
        # prompt_delta intentionally unused — SDXL doesn't use FLUX's
        # individually-tuned per-view deltas, it builds a fixed structure:
        # "(view) view, (character prompt), full body, T-pose, neutral expression"
        return f"{view_phrase}, {base_prompt}, {POSE_SUFFIX}"

    def generate(
        self,
        prompt: str,
        reference_images: list[Image.Image] | None = None,
        negative_prompt_prepend: str | None = None,
        negative_prompt_append: str | None = None,
        seed: int | None = None,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> tuple[Image.Image, int]:
        # reference_images intentionally unused — this backend has no
        # image-conditioning input. Consistency across views comes from
        # ConsistencyController passing the same `seed` on every call
        # (see consistency_strategy = "shared_seed"), not from images.
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        negative_prompt = compose_negative_prompt(
            default=self.default_negative_prompt,
            extra_prepend=negative_prompt_prepend,
            extra_append=negative_prompt_append,
        )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        return image, seed
