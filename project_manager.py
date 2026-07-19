"""
RefSheetArtist — Project / Export layer

Saves a full character project to disk (profile, per-view images +
generation metadata, composited sheet) and can reload a saved project
later so individual views can be regenerated without re-running the
whole pipeline.

Decoupled the same way compositor.py is: duck-types on `.image`,
`.prompt`, `.seed`, `.reference_view_names`, so it works with
ConsistencyController's GeneratedView objects OR this module's own
lightweight ProjectView, without importing the pipeline module.

Project folder layout:
    projects/<character_name>/
        profile.json      - CharacterProfile
        manifest.json     - per-view prompt/seed/reference metadata
        views/
            front.png
            side.png
            three_quarter.png
            back.png
        sheet.png          - composited reference sheet (if provided)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

PROFILE_FILENAME = "profile.json"
MANIFEST_FILENAME = "manifest.json"
VIEWS_DIRNAME = "views"
SHEET_FILENAME = "sheet.png"


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass
class CharacterProfile:
    """Minimal profile for now — matches where the Description Parser's
    eventual structured output should land. `extra` catches any additional
    fields added later without breaking projects saved under an older schema."""
    name: str
    base_prompt: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectView:
    """Generic, serializable stand-in for ConsistencyController.GeneratedView.
    Has the same shape (name/image/prompt/seed/reference_view_names), so a
    loaded project's views can be dropped straight into
    `ConsistencyController.views` to resume regeneration."""
    name: str
    image: Image.Image
    prompt: str
    seed: int
    reference_view_names: list[str]


def _view_to_record(view: Any) -> dict:
    """Duck-typed extraction — works for GeneratedView or ProjectView alike,
    or any object with these four attributes."""
    return {
        "prompt": getattr(view, "prompt", None),
        "seed": getattr(view, "seed", None),
        "reference_view_names": getattr(view, "reference_view_names", []),
    }


# --------------------------------------------------------------------------
# Project manager
# --------------------------------------------------------------------------

class ProjectManager:
    def __init__(self, projects_root: Path):
        self.projects_root = projects_root

    def project_dir(self, character_name: str) -> Path:
        safe_name = character_name.strip().replace(" ", "_")
        return self.projects_root / safe_name

    # -- Save -----------------------------------------------------------------

    def save_project(
        self,
        profile: CharacterProfile,
        views: dict[str, Any],           # name -> GeneratedView-like, or plain PIL.Image
        sheet_image: Image.Image | None = None,
    ) -> Path:
        proj_dir = self.project_dir(profile.name)
        views_dir = proj_dir / VIEWS_DIRNAME
        views_dir.mkdir(parents=True, exist_ok=True)

        (proj_dir / PROFILE_FILENAME).write_text(json.dumps(asdict(profile), indent=2))

        manifest: dict[str, dict] = {}
        for name, view in views.items():
            image = getattr(view, "image", view)
            img_path = views_dir / f"{name}.png"
            image.save(img_path)
            record = _view_to_record(view)
            record["image_file"] = f"{VIEWS_DIRNAME}/{img_path.name}"
            manifest[name] = record
        (proj_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

        if sheet_image is not None:
            sheet_image.save(proj_dir / SHEET_FILENAME)

        return proj_dir

    # -- Load -----------------------------------------------------------------

    def load_project(self, character_name: str) -> tuple[CharacterProfile, dict[str, ProjectView]]:
        proj_dir = self.project_dir(character_name)
        if not proj_dir.exists():
            raise FileNotFoundError(f"No project found for '{character_name}' at {proj_dir}")

        profile_data = json.loads((proj_dir / PROFILE_FILENAME).read_text())
        profile = CharacterProfile(**profile_data)

        manifest = json.loads((proj_dir / MANIFEST_FILENAME).read_text())
        views: dict[str, ProjectView] = {}
        for name, record in manifest.items():
            image = Image.open(proj_dir / record["image_file"])
            image.load()  # force full read now, since we might resave to the same path later
            views[name] = ProjectView(
                name=name,
                image=image,
                prompt=record["prompt"],
                seed=record["seed"],
                reference_view_names=record.get("reference_view_names", []),
            )
        return profile, views

    # -- Convenience ------------------------------------------------------------

    def list_projects(self) -> list[str]:
        if not self.projects_root.exists():
            return []
        return sorted(p.name for p in self.projects_root.iterdir() if p.is_dir())

    def project_exists(self, character_name: str) -> bool:
        return self.project_dir(character_name).exists()


# --------------------------------------------------------------------------
# Smoke test — round-trip save/load with placeholder images, no model required
# --------------------------------------------------------------------------

if __name__ == "__main__":
    manager = ProjectManager(projects_root=Path("./projects"))

    profile = CharacterProfile(
        name="Test Character",
        base_prompt=(
            "a stocky orange tabby cat-person blacksmith, exactly one scar, "
            "located only on the upper left arm, no other scars or wounds "
            "anywhere on the body, empty paws, holding nothing, no weapons, "
            "no tools, wears a leather apron, semi-realistic style"
        ),
    )

    fake_views = {
        name: ProjectView(
            name=name,
            image=Image.new("RGB", (256, 256), color),
            prompt=f"{profile.base_prompt}, {name} view",
            seed=42,
            reference_view_names=refs,
        )
        for name, color, refs in [
            ("front", (200, 160, 120), []),
            ("side", (160, 200, 120), ["front"]),
            ("three_quarter", (120, 160, 200), ["front", "side"]),
            ("back", (200, 120, 160), ["front", "side"]),
        ]
    }

    saved_dir = manager.save_project(profile, fake_views)
    print(f"Saved project to {saved_dir}")

    loaded_profile, loaded_views = manager.load_project("Test Character")
    assert loaded_profile.name == profile.name
    assert loaded_profile.base_prompt == profile.base_prompt
    assert set(loaded_views.keys()) == set(fake_views.keys())
    assert loaded_views["three_quarter"].reference_view_names == ["front", "side"]
    assert loaded_views["front"].image.size == (256, 256)

    print("Round-trip OK:", manager.list_projects())
