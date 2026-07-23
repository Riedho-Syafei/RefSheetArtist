"""
RefSheetArtist — Generation Backend interface

Formalizes the interface ConsistencyController relies on, so multiple
model backends (FLUX.2 Klein distilled, a furry SDXL checkpoint, whatever
comes next) can be swapped without touching ConsistencyController, the
Compositor, or the Project/Export layer at all.

Two things turned out to be genuinely backend-specific, not just
generate()'s internals:

1. Prompt structure. FLUX.2 Klein's per-view prompts were hand-tuned as
   full strings (see DEFAULT_VIEW_PLAN). A tag-based SDXL checkpoint
   needs the view name FIRST, then the character prompt, then a fixed
   pose suffix — a different order entirely, not just different words.
   compose_prompt() lets each backend arrange the pieces it's given
   however its model wants them, ignoring what it doesn't need.

2. Cross-view consistency mechanism. FLUX.2 Klein has native
   multi-reference image editing, so consistency comes from passing
   prior views as conditioning images (see reference_views on ViewSpec).
   A plain txt2img SDXL backend has no image-conditioning input at all —
   consistency there comes entirely from reusing the SAME seed across
   every view. consistency_strategy tells ConsistencyController which
   mode a backend needs so it can drive the right behavior without
   hardcoding either backend's approach.

Only put things here that EVERY backend must expose. Backend-specific
tuning (step counts, guidance scale, model paths) stays inside each
backend's own module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from PIL import Image


class GenerationBackend(ABC):
    """Common contract every image-generation backend must implement."""

    #: How many reference/conditioning images this backend can usefully
    #: accept for a single generation call. 0 for backends with no
    #: image-conditioning input at all (e.g. plain txt2img SDXL).
    max_reference_images: int = 0

    #: "reference_images" — this backend takes prior views as conditioning
    #:   images; ConsistencyController builds a reference set from
    #:   ViewSpec.reference_views and lets each view get its own seed.
    #: "shared_seed" — this backend has no image-conditioning input;
    #:   ConsistencyController ignores reference_views entirely and forces
    #:   every view to reuse the seed locked at the canonical stage.
    consistency_strategy: Literal["reference_images", "shared_seed"] = "reference_images"

    @abstractmethod
    def compose_prompt(self, base_prompt: str, view_phrase: str, prompt_delta: str) -> str:
        """Build the final prompt string for one view from generic pieces.

        base_prompt: the character's own prompt (unchanged across views).
        view_phrase: short view name, e.g. "front view", "three-quarter view".
        prompt_delta: FLUX's fuller, individually-tuned per-view descriptor
            (see DEFAULT_VIEW_PLAN) — may be ignored by backends that build
            their own fixed structure instead (e.g. SDXL).

        A backend is free to ignore whichever piece(s) don't fit its
        prompt conventions.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reference_images: list[Image.Image] | None = None,
        negative_prompt_prepend: str | None = None,
        negative_prompt_append: str | None = None,
        seed: int | None = None,
    ) -> tuple[Image.Image, int]:
        """Generate one image. Must return (image, seed_used).

        reference_images: ignored by "shared_seed" backends — they have no
        image-conditioning input, so ConsistencyController won't even
        build this list for them.

        negative_prompt_prepend / negative_prompt_append: caller-supplied
        text to add before/after this backend's own default negative
        prompt (if it has one). The backend owns loading and composing
        its default. Backends without a real negative-prompt input (e.g.
        FLUX.2 Klein) should silently ignore both.

        seed: for "shared_seed" backends, ConsistencyController will pass
        the same seed on every call after the canonical view is locked —
        the backend just needs to honor it deterministically, not
        generate a fresh one.
        """
        raise NotImplementedError
