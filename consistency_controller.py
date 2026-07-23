"""
RefSheetArtist — Consistency Controller (backend-agnostic)

Backend-agnostic in the sense that matters: this module never imports a
specific model pipeline. It DOES know that backends come in two flavors
via GenerationBackend.consistency_strategy, and branches on that —
because the two flavors genuinely need different orchestration, not just
different generate() internals:

- "reference_images" (e.g. FluxKleinBackend): canonical view first, then
  each subsequent view is generated using prior views as conditioning
  images. Each view can get its own (fresh, random) seed — the reference
  images are what keeps the character consistent.

- "shared_seed" (e.g. SDXLFurryBackend): no image-conditioning input at
  all. The canonical view's seed is locked and then reused, unchanged,
  for every other view in the project — the seed IS the consistency
  mechanism, so reference_views is never consulted for these backends.

Swap FluxKleinBackend for SDXLFurryBackend (or anything else implementing
GenerationBackend) without touching this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from generation_backend import GenerationBackend

CANONICAL_CANDIDATES = 3         # cheap now at ~1 min/image on FLUX; revisit per-backend if SDXL is much slower


@dataclass
class ViewSpec:
    """One view to generate, e.g. front / side / back / three_quarter."""
    name: str
    view_phrase: str             # short view name, e.g. "front view" — used by backends that prefix it (SDXL)
    prompt_delta: str            # fuller, individually-tuned per-view text — used by backends that append it (FLUX)
    reference_views: list[str] = field(default_factory=list)  # only consulted by "reference_images" backends
    # Extra text to add before/after a backend's own default negative prompt
    # (e.g. SDXLFurryBackend's config/sdxl_negative_prompt.txt). Backends
    # without a real negative prompt (FLUX.2 Klein) ignore both.
    negative_prompt_prepend: str | None = None
    negative_prompt_append: str | None = None


DEFAULT_VIEW_PLAN: list[ViewSpec] = [
    ViewSpec(name="front", view_phrase="front view",
              prompt_delta="front view, full body, T-pose, neutral expression"),
    ViewSpec(name="side", view_phrase="side view",
              prompt_delta="side profile view, full body, same pose",
              reference_views=["front"]),
    ViewSpec(name="three_quarter", view_phrase="three-quarter view",
              prompt_delta="three-quarter view, full body",
              reference_views=["front", "side"]),
    ViewSpec(name="back", view_phrase="back view",
              prompt_delta="back view, full body",
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
# Consistency Controller
# --------------------------------------------------------------------------

class ConsistencyController:
    def __init__(self, backend: GenerationBackend, base_character_prompt: str):
        self.backend = backend
        self.base_prompt = base_character_prompt
        self.views: dict[str, GeneratedView] = {}
        # Set by lock_canonical(). "shared_seed" backends force every
        # subsequent view to reuse this — it's the entire consistency
        # mechanism for those backends.
        self.canonical_seed: int | None = None

    # -- Stage 1: canonical view -------------------------------------------------

    def generate_canonical(
        self,
        view_spec: ViewSpec,
        num_candidates: int = CANONICAL_CANDIDATES,
    ) -> list[GeneratedView]:
        """Generate N candidate canonical images, each with its own random
        seed so the caller has genuinely different options to pick from.
        Caller (UI or auto-scorer) picks one via `lock_canonical`."""
        candidates = []
        prompt = self.backend.compose_prompt(self.base_prompt, view_spec.view_phrase, view_spec.prompt_delta)
        for _ in range(num_candidates):
            image, seed = self.backend.generate(
                prompt=prompt,
                negative_prompt_prepend=view_spec.negative_prompt_prepend,
                negative_prompt_append=view_spec.negative_prompt_append,
            )
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
        self.canonical_seed = chosen.seed

    # -- Stage 2: subsequent views -------------------------------------------------

    def _build_reference_set(self, view_spec: ViewSpec) -> list[Image.Image]:
        names = view_spec.reference_views[: self.backend.max_reference_images]
        missing = [n for n in names if n not in self.views]
        if missing:
            raise ValueError(
                f"Cannot generate '{view_spec.name}': missing prerequisite view(s) {missing}"
            )
        return [self.views[n].image for n in names]

    def generate_view(self, view_spec: ViewSpec, seed: int | None = None) -> GeneratedView:
        prompt = self.backend.compose_prompt(self.base_prompt, view_spec.view_phrase, view_spec.prompt_delta)

        if self.backend.consistency_strategy == "reference_images":
            reference_images = self._build_reference_set(view_spec)
            use_seed = seed  # None is fine here — fresh randomness per view is expected;
                              # the reference images are what keeps the character consistent.

        elif self.backend.consistency_strategy == "shared_seed":
            if self.canonical_seed is None:
                raise ValueError(
                    "No canonical seed locked yet. Call generate_canonical() + "
                    "lock_canonical() before generating other views with a "
                    "shared_seed backend — the seed IS the consistency mechanism."
                )
            reference_images = None  # this backend has no image-conditioning input; never consulted
            use_seed = seed if seed is not None else self.canonical_seed

        else:
            raise ValueError(f"Unknown consistency_strategy: {self.backend.consistency_strategy!r}")

        image, used_seed = self.backend.generate(
            prompt=prompt,
            reference_images=reference_images,
            negative_prompt_prepend=view_spec.negative_prompt_prepend,
            negative_prompt_append=view_spec.negative_prompt_append,
            seed=use_seed,
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
        """Re-run a single view, same reference set (if applicable).

        Behavior differs by backend, both correctly, via the same
        seed=None call: "reference_images" backends get a fresh random
        seed for natural variation (consistency comes from the reference
        images, not the seed). "shared_seed" backends automatically reuse
        canonical_seed instead (generate_view's default), since a
        different seed there would produce a differently-looking
        character in just that one view — not a "fix", a mismatch.
        """
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

def run_full_sheet(backend: GenerationBackend, base_character_prompt: str, out_dir: Path) -> ConsistencyController:
    controller = ConsistencyController(backend, base_character_prompt)

    front_spec = DEFAULT_VIEW_PLAN[0]
    candidates = controller.generate_canonical(front_spec)
    # Placeholder selection logic — swap for UI picker or auto-scorer.
    controller.lock_canonical(candidates[0])

    for view_spec in DEFAULT_VIEW_PLAN[1:]:
        controller.generate_view(view_spec)

    controller.save_project(out_dir)
    return controller


if __name__ == "__main__":
    from flux_klein_backend import FluxKleinBackend

    run_full_sheet(
        backend=FluxKleinBackend(),
        base_character_prompt=(
            "a stocky orange tabby cat-person blacksmith, exactly one scar, located "
            "only on the upper left arm, no other scars or wounds anywhere on the body, "
            "empty paws, holding nothing, no weapons, no tools, "
            "wears a leather apron, semi-realistic style"
        ),
        out_dir=Path("./output/test_character"),
    )
