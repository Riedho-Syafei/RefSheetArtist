"""
RefSheetArtist — Canonical candidate picker

Replaces the `candidates[0]` placeholder with a real interactive picker:
saves every candidate image to disk, opens each one (best-effort, OS
default viewer), and asks which one to lock in as the canonical view.

Decoupled the same way as compositor.py / project_manager.py: duck-types
on `.image`, never imports consistency_controller.py.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Sequence


def save_candidates(
    candidates: Sequence[Any], out_dir: Path, view_name: str = "front"
) -> list[Path]:
    """Saves each candidate's image to out_dir as {view_name}_candidate_{n}.png.
    Returns saved paths in the same order as `candidates`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, candidate in enumerate(candidates, start=1):
        image = getattr(candidate, "image", candidate)
        path = out_dir / f"{view_name}_candidate_{i}.png"
        image.save(path)
        paths.append(path)
    return paths


def _open_in_viewer(path: Path) -> None:
    """Best-effort open in the OS's default image viewer. Silently does
    nothing on failure — worst case you open the file yourself."""
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def pick_canonical_interactive(
    candidates: Sequence[Any],
    out_dir: Path,
    view_name: str = "front",
    auto_open: bool = True,
) -> Any:
    """Saves all candidates to disk, opens them (best-effort), prompts the
    user to choose by number, and returns the chosen candidate object
    unchanged (the same object the caller passed in — e.g. a GeneratedView)."""
    if not candidates:
        raise ValueError("No candidates to choose from.")

    paths = save_candidates(candidates, out_dir, view_name)

    print(f"\nGenerated {len(candidates)} candidate(s) for '{view_name}':")
    for i, path in enumerate(paths, start=1):
        print(f"  [{i}] {path}")
        if auto_open:
            _open_in_viewer(path)

    while True:
        choice = input(f"Pick the canonical '{view_name}' image (1-{len(candidates)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            print(f"Locked in candidate {choice} as canonical.\n")
            return candidates[int(choice) - 1]
        print("Invalid choice, try again.")
