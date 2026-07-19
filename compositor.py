"""
RefSheetArtist — Compositor

Assembles generated character views into a single reference sheet image:
2x2 grid, each view labeled, simple header with the character's name.

Deliberately decoupled from ConsistencyController — it accepts plain PIL
images (or anything with an `.image` attribute, so a dict of GeneratedView
objects works too via `compose_from_controller`), so it doesn't need to
import or touch the generation pipeline at all.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Config — tweak freely, nothing below depends on exact values
# --------------------------------------------------------------------------

CELL_SIZE = (1024, 1024)     # each view's thumbnail box (w, h)
CELL_PADDING = 24          # gap between cells
LABEL_HEIGHT = 40          # space reserved under each cell for its label
HEADER_HEIGHT = 80         # space reserved at top for character name
MARGIN = 32                # outer margin around the whole sheet

BACKGROUND_COLOR = (255, 255, 255)
LABEL_COLOR = (40, 40, 40)
HEADER_COLOR = (20, 20, 20)

DEFAULT_VIEW_ORDER = ["front", "side", "three_quarter", "back"]
DEFAULT_VIEW_LABELS = {
    "front": "Front",
    "side": "Side",
    "three_quarter": "3/4 View",
    "back": "Back",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try for a real TTF font; fall back to PIL's built-in bitmap font
    if none is available on this machine (still renders fine, just plainer)."""
    for candidate in ("DejaVuSans-Bold.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_image_to_cell(image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    """Resize `image` to fit inside `cell_size` preserving aspect ratio,
    centered on a cell-sized canvas so all cells line up regardless of the
    source image's shape."""
    fitted = image.convert("RGB").copy()
    fitted.thumbnail(cell_size, Image.LANCZOS)
    canvas = Image.new("RGB", cell_size, BACKGROUND_COLOR)
    offset = ((cell_size[0] - fitted.width) // 2, (cell_size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: int,
    box_top: int,
    box_height: int,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = center_x - text_w // 2
    y = box_top + (box_height - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=fill, font=font)


# --------------------------------------------------------------------------
# Core compositor
# --------------------------------------------------------------------------

def compose_reference_sheet(
    views: dict[str, Image.Image],
    character_name: str,
    out_path: Path,
    view_order: list[str] | None = None,
    view_labels: dict[str, str] | None = None,
) -> Image.Image:
    """
    views: mapping of view name -> PIL Image (already-generated views).
    Raises ValueError if any view in `view_order` is missing, so a
    half-finished sheet never silently gets produced.
    """
    view_order = view_order or DEFAULT_VIEW_ORDER
    view_labels = view_labels or DEFAULT_VIEW_LABELS

    missing = [v for v in view_order if v not in views]
    if missing:
        raise ValueError(f"Missing view(s) for composite: {missing}")

    cols, rows = 2, 2
    cell_w, cell_h = CELL_SIZE
    grid_w = cols * cell_w + (cols - 1) * CELL_PADDING
    grid_h = rows * (cell_h + LABEL_HEIGHT) + (rows - 1) * CELL_PADDING

    sheet_w = grid_w + 2 * MARGIN
    sheet_h = HEADER_HEIGHT + grid_h + 2 * MARGIN

    sheet = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(sheet)

    # Header
    header_font = _load_font(36)
    _draw_centered_text(
        draw, character_name, header_font,
        center_x=sheet_w // 2, box_top=0, box_height=HEADER_HEIGHT,
        fill=HEADER_COLOR,
    )

    # Grid cells
    label_font = _load_font(24)
    for idx, view_name in enumerate(view_order):
        row, col = divmod(idx, cols)
        cell_x = MARGIN + col * (cell_w + CELL_PADDING)
        cell_y = HEADER_HEIGHT + MARGIN + row * (cell_h + LABEL_HEIGHT + CELL_PADDING)

        fitted = _fit_image_to_cell(views[view_name], CELL_SIZE)
        sheet.paste(fitted, (cell_x, cell_y))

        label = view_labels.get(view_name, view_name.replace("_", " ").title())
        _draw_centered_text(
            draw, label, label_font,
            center_x=cell_x + cell_w // 2, box_top=cell_y + cell_h, box_height=LABEL_HEIGHT,
            fill=LABEL_COLOR,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return sheet


# --------------------------------------------------------------------------
# Adapter: build directly from a ConsistencyController's `.views` dict
# --------------------------------------------------------------------------

def compose_from_controller(
    controller_views: dict[str, object],
    character_name: str,
    out_path: Path,
    view_order: list[str] | None = None,
    view_labels: dict[str, str] | None = None,
) -> Image.Image:
    """
    controller_views: `ConsistencyController.views`, i.e. dict[str, GeneratedView].
    Duck-types on `.image` so it works with GeneratedView or a plain dict of
    PIL Images alike — this file never imports consistency_controller.py,
    so it doesn't need the model/pipeline loaded to run.
    """
    plain_images = {
        name: (getattr(gv, "image", gv)) for name, gv in controller_views.items()
    }
    return compose_reference_sheet(
        plain_images, character_name, out_path, view_order, view_labels
    )


if __name__ == "__main__":
    views = {
        "front": Image.open("output/test_character/front.png"),
        "side": Image.open("output/test_character/side.png"),
        "three_quarter": Image.open("output/test_character/three_quarter.png"),
        "back": Image.open("output/test_character/back.png"),
    }
    compose_reference_sheet(views, "Orange Tabby Blacksmith", Path("./output/sheet.png"))