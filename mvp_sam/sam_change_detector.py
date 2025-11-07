"""SAM-driven change detection pipeline.

This module wires the Segment Anything Model (SAM) into a simple change-detection
workflow tailored for before/after satellite imagery. It segments the two
images, matches their masks, and decides whether an object is new, removed, or
modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from .sam_patches import ensure_float32_mps_patch


@dataclass
class MaskItem:
    """Container with the subset of SAM mask metadata we care about."""

    mask_id: str
    segmentation: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    score: float
    source: str  # either "before" or "after"

    @property
    def centroid(self) -> Tuple[float, float]:
        rows, cols = np.nonzero(self.segmentation)
        if len(rows) == 0:
            return 0.0, 0.0
        return float(cols.mean()), float(rows.mean())


class SAMChangeDetector:
    """Encapsulates SAM inference and simple mask-based change detection."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: Optional[str] = None,
        mask_generator_params: Optional[Dict] = None,
        match_iou_threshold: float = 0.45,
        modification_area_threshold: float = 0.15,
        modification_shift_threshold: float = 0.05,
    ) -> None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at '{checkpoint_path}'. See README for setup instructions."
            )
        if device is None:
            device = self._infer_device()

        if model_type not in sam_model_registry:
            raise ValueError(
                f"Unknown SAM model '{model_type}'. Valid options: {', '.join(sam_model_registry.keys())}."
            )

        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()

        generator_defaults = dict(
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=256,
        )
        if mask_generator_params:
            generator_defaults.update(mask_generator_params)

        ensure_float32_mps_patch()
        self.mask_generator = SamAutomaticMaskGenerator(self.model, **generator_defaults)
        self.match_iou_threshold = match_iou_threshold
        self.modification_area_threshold = modification_area_threshold
        self.modification_shift_threshold = modification_shift_threshold

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image cannot be None")
        arr = image.copy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr

    def segment(self, image: np.ndarray, source: str) -> List[MaskItem]:
        prepared = self._ensure_rgb_uint8(image)
        raw_masks = self.mask_generator.generate(prepared)
        return [
            MaskItem(
                mask_id=f"{source}_{idx}",
                segmentation=item["segmentation"].astype(bool),
                bbox=tuple(int(v) for v in item["bbox"]),
                area=int(item["area"]),
                score=float(item.get("predicted_iou", 0.0)),
                source=source,
            )
            for idx, item in enumerate(raw_masks)
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def _match_masks(
        self, before_masks: List[MaskItem], after_masks: List[MaskItem]
    ) -> Tuple[Dict[str, Tuple[MaskItem, float]], List[str], List[str]]:
        matches: Dict[str, Tuple[MaskItem, float]] = {}
        matched_after_ids: set[str] = set()

        for before in before_masks:
            best_iou = 0.0
            best_after: Optional[MaskItem] = None
            for after in after_masks:
                if after.mask_id in matched_after_ids:
                    continue
                iou = self._compute_iou(before.segmentation, after.segmentation)
                if iou > best_iou:
                    best_iou = iou
                    best_after = after
            if best_after and best_iou >= self.match_iou_threshold:
                matches[before.mask_id] = (best_after, best_iou)
                matched_after_ids.add(best_after.mask_id)

        unmatched_before = [m.mask_id for m in before_masks if m.mask_id not in matches]
        unmatched_after = [m.mask_id for m in after_masks if m.mask_id not in matched_after_ids]
        return matches, unmatched_before, unmatched_after

    # ------------------------------------------------------------------
    def detect_changes(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        min_area: int = 0,
    ) -> Dict[str, List]:
        before_rgb, after_rgb = self._align_inputs(img_before, img_after)
        masks_before = self.segment(before_rgb, source="before")
        masks_after = self.segment(after_rgb, source="after")

        if min_area > 0:
            masks_before = [m for m in masks_before if m.area >= min_area]
            masks_after = [m for m in masks_after if m.area >= min_area]
        before_lookup = {mask.mask_id: mask for mask in masks_before}
        after_lookup = {mask.mask_id: mask for mask in masks_after}

        matches, removed_ids, new_ids = self._match_masks(masks_before, masks_after)

        removed = [m for m in masks_before if m.mask_id in removed_ids]
        new = [m for m in masks_after if m.mask_id in new_ids]
        modified = []
        unchanged = []

        for before_id, (after_mask, iou) in matches.items():
            before_mask = before_lookup[before_id]
            area_delta = abs(before_mask.area - after_mask.area) / max(before_mask.area, after_mask.area)
            centroid_delta = self._centroid_shift(before_mask, after_mask, before_rgb.shape[:2])
            if (
                area_delta >= self.modification_area_threshold
                or centroid_delta >= self.modification_shift_threshold
            ):
                modified.append(
                    {
                        "before": before_mask,
                        "after": after_mask,
                        "iou": iou,
                        "area_delta": area_delta,
                        "centroid_shift": centroid_delta,
                    }
                )
            else:
                unchanged.append((before_mask, after_mask, iou))

        return {
            "masks_before": masks_before,
            "masks_after": masks_after,
            "new": new,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
            "before_image": before_rgb,
            "after_image": after_rgb,
        }

    # ------------------------------------------------------------------
    def _align_inputs(self, img_before: np.ndarray, img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        before = self._ensure_rgb_uint8(img_before)
        after = self._ensure_rgb_uint8(img_after)
        if before.shape[:2] != after.shape[:2]:
            after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_LINEAR)
        return before, after

    def _centroid_shift(self, before: MaskItem, after: MaskItem, image_shape: Tuple[int, int]) -> float:
        before_centroid = np.array(before.centroid)
        after_centroid = np.array(after.centroid)
        height, width = image_shape
        diagonal = np.sqrt(height**2 + width**2)
        if diagonal == 0:
            return 0.0
        return float(np.linalg.norm(after_centroid - before_centroid) / diagonal)
