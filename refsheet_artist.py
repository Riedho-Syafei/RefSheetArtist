"""
RefSheetArtist — End-to-end run script

Chains everything built so far into one flow:

    CharacterProfile (name + prompt)
          v
    GenerationBackend       (FluxKleinBackend or SDXLFurryBackend)
          v
    ConsistencyController   (canonical + reference-chained views)
          v
    Compositor              (2x2 labeled reference sheet)
          v
    ProjectManager          (saves profile/views/manifest/sheet to disk)

Backend selection is now a CLI flag (--backend flux | sdxl) instead
of hardcoded — ConsistencyController, Compositor, and ProjectManager don't
know or care which one is active.

Supports three modes via CLI flags:
  1. Fresh generation of a new (or force-regenerated) character.
  2. Loading an already-saved project and just re-compositing the sheet.
  3. Regenerating a single view of an existing project, then re-compositing
     and re-saving — the cheap "fix one bad view" path.

NOTE: backend imports for whichever backend ISN'T selected are deferred
(imported inside the functions that need them), so e.g. selecting --backend
flux doesn't require diffusers' SDXL pipeline classes to import cleanly,
and vice versa.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canonical_picker import pick_canonical_interactive
from compositor import compose_reference_sheet
from consistency_controller import (
    DEFAULT_VIEW_PLAN,
    ConsistencyController,
    ViewSpec,
)
from generation_backend import GenerationBackend
from project_manager import CharacterProfile, ProjectManager

PROJECTS_ROOT = Path("./projects")
SHEET_FILENAME = "sheet.png"
BACKEND_CHOICES = ["flux", "sdxl"]


def _neg_prompt_view_plan(
    view_plan: list[ViewSpec],
    prepend: str | None,
    append: str | None,
) -> list[ViewSpec]:
    """Return a copy of *view_plan* with *prepend* and/or *append* injected
    into every view's negative_prompt_prepend / negative_prompt_append fields.

    If both are None the original list is returned unchanged (no copy).
    """
    if prepend is None and append is None:
        return view_plan
    return [
        ViewSpec(
            name=v.name,
            view_phrase=v.view_phrase,
            prompt_delta=v.prompt_delta,
            reference_views=list(v.reference_views),
            negative_prompt_prepend=prepend,
            negative_prompt_append=append,
        )
        for v in view_plan
    ]


def _build_backend(name: str) -> GenerationBackend:
    if name == "flux":
        from flux_klein_backend import FluxKleinBackend
        return FluxKleinBackend()
    if name == "sdxl":
        from sdxl_backend import SDXLBackend
        return SDXLBackend()
    raise ValueError(f"Unknown backend '{name}'. Choose from {BACKEND_CHOICES}.")


def _sheet_path_for(manager: ProjectManager, character_name: str) -> Path:
    return manager.project_dir(character_name) / SHEET_FILENAME


def generate_full_sheet(
    character_name: str,
    base_prompt: str,
    manager: ProjectManager,
    backend_name: str,
    view_plan: list[ViewSpec] = DEFAULT_VIEW_PLAN,
    interactive_pick: bool = True,
) -> None:
    """Runs the full pipeline for a new character: canonical -> reference
    chain -> composite -> save.

    interactive_pick: when True (default), saves all canonical candidates
    to disk, opens them, and asks you to choose one. When False, picks the
    first candidate automatically — useful for scripted/automated runs
    (e.g. tests) where nothing can respond to an input() prompt.
    """
    backend = _build_backend(backend_name)
    controller = ConsistencyController(backend, base_prompt)

    front_spec = view_plan[0]
    candidates = controller.generate_canonical(front_spec)

    if interactive_pick:
        candidates_dir = manager.project_dir(character_name) / "candidates"
        chosen = pick_canonical_interactive(candidates, out_dir=candidates_dir, view_name=front_spec.name)
    else:
        chosen = candidates[0]
    controller.lock_canonical(chosen)

    for view_spec in view_plan[1:]:
        controller.generate_view(view_spec)

    sheet_image = compose_reference_sheet(
        {name: gv.image for name, gv in controller.views.items()},
        character_name=character_name,
        out_path=_sheet_path_for(manager, character_name),
    )

    profile = CharacterProfile(name=character_name, base_prompt=base_prompt, backend=backend_name)
    manager.save_project(profile, controller.views, sheet_image)
    print(f"Saved '{character_name}' to {manager.project_dir(character_name)}")


def regenerate_single_view(
    character_name: str,
    view_name: str,
    manager: ProjectManager,
    view_plan: list[ViewSpec] = DEFAULT_VIEW_PLAN,
) -> None:
    """Loads an existing project, regenerates one view against its existing
    reference set, re-composites, and re-saves. Doesn't touch the other views.

    Uses the backend the project was originally generated with (stored on
    the profile) so a regenerated view can't silently come from a
    different model than the rest of the sheet.
    """
    profile, loaded_views = manager.load_project(character_name)

    backend = _build_backend(profile.backend)
    controller = ConsistencyController(backend, profile.base_prompt)
    controller.views = loaded_views  # loaded ProjectView objects drop in directly

    controller.regenerate_view(view_name, view_plan)

    sheet_image = compose_reference_sheet(
        {name: gv.image for name, gv in controller.views.items()},
        character_name=character_name,
        out_path=_sheet_path_for(manager, character_name),
    )
    manager.save_project(profile, controller.views, sheet_image)
    print(f"Regenerated '{view_name}' for '{character_name}' and re-saved.")


def recomposite_only(character_name: str, manager: ProjectManager) -> None:
    """Loads an existing project and just re-runs the compositor — no
    generation at all. Useful after changing compositor.py settings
    (cell size, layout, labels) without wanting to regenerate images."""
    profile, loaded_views = manager.load_project(character_name)
    sheet_image = compose_reference_sheet(
        {name: v.image for name, v in loaded_views.items()},
        character_name=character_name,
        out_path=_sheet_path_for(manager, character_name),
    )
    manager.save_project(profile, loaded_views, sheet_image)
    print(f"Re-composited sheet for '{character_name}'.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RefSheetArtist end-to-end run")
    parser.add_argument("--name", required=True, help="Character name / project name")
    parser.add_argument("--prompt", help="Base character prompt (required for fresh generation)")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="flux",
        help="Which generation backend to use for fresh generation (default: flux)",
    )
    parser.add_argument("--projects-dir", default=str(PROJECTS_ROOT))
    parser.add_argument(
        "--regenerate-view",
        help="Regenerate only this view (e.g. 'side') against an existing saved project",
    )
    parser.add_argument(
        "--recomposite-only",
        action="store_true",
        help="Skip generation entirely, just rebuild the sheet from saved views",
    )
    parser.add_argument(
        "--no-interactive-pick",
        action="store_true",
        help="Auto-pick the first canonical candidate instead of asking you to choose",
    )
    parser.add_argument(
        "--negative-prompt-prepend",
        help="Extra negative prompt text inserted BEFORE the backend's default "
             "(applied to every view; FLUX backend silently ignores it)",
    )
    parser.add_argument(
        "--negative-prompt-append",
        help="Extra negative prompt text inserted AFTER the backend's default "
             "(applied to every view; FLUX backend silently ignores it)",
    )
    args = parser.parse_args()

    manager = ProjectManager(projects_root=Path(args.projects_dir))

    if args.recomposite_only:
        recomposite_only(args.name, manager)
        return

    # Build a view plan with the caller's negative-prompt extras if given.
    view_plan = _neg_prompt_view_plan(
        DEFAULT_VIEW_PLAN,
        prepend=args.negative_prompt_prepend,
        append=args.negative_prompt_append,
    )

    if args.regenerate_view:
        regenerate_single_view(args.name, args.regenerate_view, manager,
                                view_plan=view_plan)
        return

    if not args.prompt:
        parser.error("--prompt is required unless using --regenerate-view or --recomposite-only")

    generate_full_sheet(
        args.name, args.prompt, manager,
        backend_name=args.backend,
        view_plan=view_plan,
        interactive_pick=not args.no_interactive_pick,
    )


if __name__ == "__main__":
    main()
