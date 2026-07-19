"""
RefSheetArtist — End-to-end run script

Chains everything built so far into one flow:

    CharacterProfile (name + prompt)
          v
    ConsistencyController  (canonical + reference-chained views)
          v
    Compositor              (2x2 labeled reference sheet)
          v
    ProjectManager           (saves profile/views/manifest/sheet to disk)

Supports three modes via CLI flags:
  1. Fresh generation of a new (or force-regenerated) character.
  2. Loading an already-saved project and just re-compositing the sheet
     (e.g. after you manually re-ran the compositor with new settings).
  3. Regenerating a single view of an existing project, then re-compositing
     and re-saving — the cheap "fix one bad view" path.

NOTE: This script imports RefSheetPipeline, which loads the real
Flux2KleinPipeline on import-time __init__ (not on module import) — so
running this for real requires your local model path + GPU, same as
before. The wiring itself (data flow between the three components) can
be sanity-checked independently using a fake pipeline substitute; see
test_end_to_end_wiring.py alongside this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compositor import compose_reference_sheet
from consistency_controller import (
    DEFAULT_VIEW_PLAN,
    ConsistencyController,
    RefSheetPipeline,
    ViewSpec,
)
from project_manager import CharacterProfile, ProjectManager

PROJECTS_ROOT = Path("./projects")
SHEET_FILENAME = "sheet.png"


def _sheet_path_for(manager: ProjectManager, character_name: str) -> Path:
    return manager.project_dir(character_name) / SHEET_FILENAME


def generate_full_sheet(
    character_name: str,
    base_prompt: str,
    manager: ProjectManager,
    view_plan: list[ViewSpec] = DEFAULT_VIEW_PLAN,
) -> None:
    """Runs the full pipeline for a new character: canonical -> reference
    chain -> composite -> save."""
    pipeline = RefSheetPipeline()
    controller = ConsistencyController(pipeline, base_prompt)

    front_spec = view_plan[0]
    candidates = controller.generate_canonical(front_spec)
    # Placeholder selection — swap for a UI picker or auto-scorer later.
    controller.lock_canonical(candidates[0])

    for view_spec in view_plan[1:]:
        controller.generate_view(view_spec)

    sheet_image = compose_reference_sheet(
        {name: gv.image for name, gv in controller.views.items()},
        character_name=character_name,
        out_path=_sheet_path_for(manager, character_name),
    )

    profile = CharacterProfile(name=character_name, base_prompt=base_prompt)
    manager.save_project(profile, controller.views, sheet_image)
    print(f"Saved '{character_name}' to {manager.project_dir(character_name)}")


def regenerate_single_view(
    character_name: str,
    view_name: str,
    manager: ProjectManager,
    view_plan: list[ViewSpec] = DEFAULT_VIEW_PLAN,
) -> None:
    """Loads an existing project, regenerates one view against its existing
    reference set, re-composites, and re-saves. Doesn't touch the other views."""
    profile, loaded_views = manager.load_project(character_name)

    pipeline = RefSheetPipeline()
    controller = ConsistencyController(pipeline, profile.base_prompt)
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
    args = parser.parse_args()

    manager = ProjectManager(projects_root=Path(args.projects_dir))

    if args.recomposite_only:
        recomposite_only(args.name, manager)
        return

    if args.regenerate_view:
        regenerate_single_view(args.name, args.regenerate_view, manager)
        return

    if not args.prompt:
        parser.error("--prompt is required unless using --regenerate-view or --recomposite-only")

    generate_full_sheet(args.name, args.prompt, manager)


if __name__ == "__main__":
    main()
