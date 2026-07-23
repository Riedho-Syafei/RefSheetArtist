"""
RefSheetArtist — SDXL default negative prompt: config loading + composition

Only meaningful for tag-based, real-negative-prompt backends (currently
SDXLBackend). FLUX.2 Klein has no real negative prompt input, so it
never touches this module.

Kept separate from sdxl_backend.py so the load/compose logic is
independently testable without needing torch/diffusers importable, and
so a future non-SDXL tag-based backend (if you ever add one) can reuse it.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "sdxl_negative_prompt.txt"


def load_default_negative_prompt(path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Read the negative-prompt config file and return its content as-is
    (whitespace-trimmed), ready to hand to an SDXL pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Default negative prompt config not found at {path}. "
            "Create it or pass a different path."
        )
    return path.read_text(encoding="utf-8").strip()


def compose_negative_prompt(
    default: str,
    extra_prepend: str | None = None,
    extra_append: str | None = None,
) -> str:
    """Combine the default negative prompt with optional caller-supplied
    extras, inserted before and/or after the default. Either, both, or
    neither extra may be given.

    Example:
        compose_negative_prompt(
            default="worst quality, bad anatomy",
            extra_prepend="extra scars",
            extra_append="wings visible from front",
        )
        -> "extra scars, worst quality, bad anatomy, wings visible from front"
    """
    parts = [
        p.strip().strip(",").strip()
        for p in (extra_prepend, default, extra_append)
        if p and p.strip()
    ]
    return ", ".join(parts)
