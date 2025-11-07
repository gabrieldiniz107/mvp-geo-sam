"""Visualization helpers for SAM-based change detection."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .sam_change_detector import MaskItem

Palette = {
    "new": (46, 204, 113),  # green
    "removed": (231, 76, 60),  # red
    "modified": (241, 196, 15),  # yellow
    "unchanged": (149, 165, 166),  # grey
}


def _to_rgb(image: np.ndarray) -> np.ndarray:
    arr = image
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def overlay_masks(image: np.ndarray, masks: Sequence[MaskItem], alpha: float = 0.5) -> np.ndarray:
    base = _to_rgb(image.astype(np.uint8).copy())
    overlay = np.zeros_like(base)
    for idx, mask in enumerate(masks):
        if mask.segmentation.shape[:2] != base.shape[:2]:
            continue
        color = _color_from_index(idx)
        overlay[mask.segmentation] = color
    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return blended


def render_change_map(
    image_shape: Tuple[int, int],
    new_masks: Sequence[MaskItem],
    removed_masks: Sequence[MaskItem],
    modified_masks: Sequence[dict],
    unchanged_masks: Sequence[Tuple[MaskItem, MaskItem, float]],
) -> np.ndarray:
    height, width = image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = Palette["unchanged"]

    _paint_masks(canvas, [entry[1] for entry in unchanged_masks], Palette["unchanged"])
    _paint_masks(canvas, [item["after"] for item in modified_masks], Palette["modified"])
    _paint_masks(canvas, new_masks, Palette["new"])
    _paint_masks(canvas, removed_masks, Palette["removed"])
    return canvas


def _paint_masks(canvas: np.ndarray, masks: Iterable[MaskItem], color: Tuple[int, int, int]) -> None:
    for mask in masks:
        if mask.segmentation.shape[:2] != canvas.shape[:2]:
            continue
        canvas[mask.segmentation] = color


def _color_from_index(idx: int) -> Tuple[int, int, int]:
    # deterministic fast palette using a few prime multipliers
    base_colors = [
        (52, 152, 219),
        (155, 89, 182),
        (26, 188, 156),
        (241, 196, 15),
        (231, 76, 60),
        (46, 204, 113),
    ]
    return base_colors[idx % len(base_colors)]
