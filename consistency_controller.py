"""
RefSheetArtist — Consistency Controller (v2, distilled-model-only)

Rough sketch of the canonical-image-then-reference-chain flow using
Flux2KleinPipeline (distilled 4B).

NOTE: This is a first-pass sketch, not verified end-to-end against your
installed diffusers version. The pipeline class name, exact kwarg names
(e.g. whether multi-reference images are passed as `image=[...]` or a
different kwarg), and step/guidance defaults for the distilled checkpoint
should be double-checked against the installed diffusers version's
Flux2KleinPipeline docstring before relying on this. Treat this as the
structural skeleton to adapt, not drop-in-final code.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DISTILLED_MODEL_ID = r"C:\AI\models\FLUX.2-klein-4B"  # distilled checkpoint
NUM_INFERENCE_STEPS = 4          # distilled model: few-step by design
MAX_REFERENCE_IMAGES = 2         # cap for 8GB VRAM + cpu offload, per design doc
CANONICAL_CANDIDATES = 3         # cheap now at ~1 min/image


@dataclass
class ViewSpec:
    """One view to generate, e.g. front / side / back / three_quarter."""
    name: str
    prompt_delta: str            # appended/merged into the base character prompt
    reference_views: list[str] = field(default_factory=list)  # which prior views to reference


DEFAULT_VIEW_PLAN: list[ViewSpec] = [
    ViewSpec(name="front", prompt_delta="front view, full body, T-pose, neutral expression"),
    ViewSpec(name="side", prompt_delta="side profile view, full body, same pose",
              reference_views=["front"]),
    ViewSpec(name="three_quarter", prompt_delta="three-quarter view, full body",
              reference_views=["front", "side"]),
    ViewSpec(name="back", prompt_delta="back view, full body",
              reference_views=["front", "side"]),
]


@dataclass
class GeneratedView:
    name: str
    image: Image.Image
    prompt: str
    seed: int
    reference_view_names: list[str]


# --------------------------------------------------------------------------
# Pipeline wrapper
# --------------------------------------------------------------------------

class RefSheetPipeline:
    """Thin wrapper around Flux2KleinPipeline for this project's use."""

    def __init__(self, model_id: str = DISTILLED_MODEL_ID, device: str = "cuda"):
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=True        # Stops it from trying to connect to the internet
        )
        # Confirmed necessary on 8GB VRAM in testing.
        self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        prompt: str,
        reference_images: list[Image.Image] | None = None,
        seed: int | None = None,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ) -> tuple[Image.Image, int]:
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
            # Verify against your diffusers version whether this is `image=` or
            # a dedicated `reference_images=` kwarg.
            kwargs["image"] = reference_images

        result = self.pipe(**kwargs)
        image = result.images[0]
        return image, seed


# --------------------------------------------------------------------------
# Consistency Controller
# --------------------------------------------------------------------------

class ConsistencyController:
    def __init__(self, pipeline: RefSheetPipeline, base_character_prompt: str):
        self.pipeline = pipeline
        self.base_prompt = base_character_prompt
        self.views: dict[str, GeneratedView] = {}

    # -- Stage 1: canonical view -------------------------------------------------

    def generate_canonical(
        self,
        view_spec: ViewSpec,
        num_candidates: int = CANONICAL_CANDIDATES,
    ) -> list[GeneratedView]:
        """Generate N candidate canonical images. Caller (UI or auto-scorer)
        picks one via `lock_canonical`."""
        candidates = []
        prompt = f"{self.base_prompt}, {view_spec.prompt_delta}"
        for _ in range(num_candidates):
            image, seed = self.pipeline.generate(prompt=prompt)
            candidates.append(
                GeneratedView(
                    name=view_spec.name,
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    reference_view_names=[],
                )
            )
        return candidates

    def lock_canonical(self, chosen: GeneratedView) -> None:
        self.views[chosen.name] = chosen

    # -- Stage 2: reference chain -------------------------------------------------

    def _build_reference_set(self, view_spec: ViewSpec) -> list[Image.Image]:
        names = view_spec.reference_views[:MAX_REFERENCE_IMAGES]
        missing = [n for n in names if n not in self.views]
        if missing:
            raise ValueError(
                f"Cannot generate '{view_spec.name}': missing prerequisite view(s) {missing}"
            )
        return [self.views[n].image for n in names]

    def generate_view(self, view_spec: ViewSpec, seed: int | None = None) -> GeneratedView:
        reference_images = self._build_reference_set(view_spec)
        prompt = f"{self.base_prompt}, {view_spec.prompt_delta}"
        image, used_seed = self.pipeline.generate(
            prompt=prompt,
            reference_images=reference_images,
            seed=seed,
        )
        generated = GeneratedView(
            name=view_spec.name,
            image=image,
            prompt=prompt,
            seed=used_seed,
            reference_view_names=view_spec.reference_views,
        )
        self.views[view_spec.name] = generated
        return generated

    # -- Stage 3: regeneration -----------------------------------------------------

    def regenerate_view(self, view_name: str, view_plan: list[ViewSpec]) -> GeneratedView:
        """Re-run a single view with a fresh seed, same reference set."""
        spec = next(v for v in view_plan if v.name == view_name)
        return self.generate_view(spec, seed=None)

    # -- Persistence ----------------------------------------------------------------

    def save_project(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, gv in self.views.items():
            img_path = out_dir / f"{name}.png"
            gv.image.save(img_path)
            manifest[name] = {
                "prompt": gv.prompt,
                "seed": gv.seed,
                "reference_view_names": gv.reference_view_names,
                "image_file": img_path.name,
            }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------
# Example run
# --------------------------------------------------------------------------

def run_full_sheet(base_character_prompt: str, out_dir: Path) -> ConsistencyController:
    pipeline = RefSheetPipeline()
    controller = ConsistencyController(pipeline, base_character_prompt)

    front_spec = DEFAULT_VIEW_PLAN[0]
    candidates = controller.generate_canonical(front_spec)
    # Placeholder selection logic — swap for UI picker or auto-scorer.
    controller.lock_canonical(candidates[0])

    for view_spec in DEFAULT_VIEW_PLAN[1:]:
        controller.generate_view(view_spec)

    controller.save_project(out_dir)
    return controller


if __name__ == "__main__":
    run_full_sheet(
        base_character_prompt=(
            "a stocky orange tabby cat-person blacksmith, exactly one scar, located "
            "only on the upper left arm, no other scars or wounds anywhere on the body, "
            "empty paws, holding nothing, no weapons, no tools, "
            "wears a leather apron, semi-realistic style"
        ),
        out_dir=Path("./output/test_character"),
    )
