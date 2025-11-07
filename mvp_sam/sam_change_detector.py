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
from segment_anything.utils.amg import build_all_layer_point_grids
from skimage.exposure import match_histograms
from skimage.metrics import hausdorff_distance

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


@dataclass
class MatchMetrics:
    """Stores similarity metrics computed during mask matching."""

    iou: float
    hausdorff_similarity: float
    score: float


class SAMChangeDetector:
    """Encapsulates SAM inference and simple mask-based change detection."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: Optional[str] = None,
        mask_generator_params: Optional[Dict] = None,
        match_iou_threshold: float = 0.45,
        match_score_threshold: float = 0.5,
        iou_weight: float = 0.65,
        hausdorff_weight: float = 0.35,
        enable_histogram_matching: bool = True,
        use_fixed_prompt_grid: bool = True,
        fixed_point_grid_size: int = 32,
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

        self.match_iou_threshold = match_iou_threshold
        self.match_score_threshold = match_score_threshold
        self.iou_weight = iou_weight
        self.hausdorff_weight = hausdorff_weight
        self.enable_histogram_matching = enable_histogram_matching
        self.use_fixed_prompt_grid = use_fixed_prompt_grid
        self.fixed_point_grid_size = fixed_point_grid_size
        self.modification_area_threshold = modification_area_threshold
        self.modification_shift_threshold = modification_shift_threshold
        self._generator_overrides = mask_generator_params or {}

        self._generator_base_params = dict(
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=256,
            crop_n_layers=0,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=1,
        )

        ensure_float32_mps_patch()
        self._normalize_match_weights()
        self._refresh_mask_generator()

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _normalize_match_weights(self) -> None:
        total = self.iou_weight + self.hausdorff_weight
        if total <= 0.0:
            self.iou_weight = 1.0
            self.hausdorff_weight = 0.0
            return
        self.iou_weight /= total
        self.hausdorff_weight /= total

    def update_prompt_grid(self, grid_size: int) -> None:
        if grid_size <= 0 or grid_size == self.fixed_point_grid_size:
            return
        self.fixed_point_grid_size = grid_size
        self._refresh_mask_generator()

    def _refresh_mask_generator(self) -> None:
        generator_params = {**self._generator_base_params}
        generator_params.update(self._generator_overrides)
        if self.use_fixed_prompt_grid:
            point_grids = build_all_layer_point_grids(
                self.fixed_point_grid_size,
                generator_params.get("crop_n_layers", 0),
                generator_params.get("crop_n_points_downscale_factor", 1),
            )
            generator_params["points_per_side"] = None
            generator_params["point_grids"] = point_grids
        self.mask_generator = SamAutomaticMaskGenerator(self.model, **generator_params)

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

    def _compute_similarity_metrics(
        self,
        before_mask: MaskItem,
        after_mask: MaskItem,
        diagonal: float,
    ) -> MatchMetrics:
        iou = self._compute_iou(before_mask.segmentation, after_mask.segmentation)
        hausdorff_sim = self._hausdorff_similarity(before_mask.segmentation, after_mask.segmentation, diagonal)
        score = (self.iou_weight * iou) + (self.hausdorff_weight * hausdorff_sim)
        return MatchMetrics(iou=iou, hausdorff_similarity=hausdorff_sim, score=score)

    def _hausdorff_similarity(self, mask_a: np.ndarray, mask_b: np.ndarray, diagonal: float) -> float:
        if diagonal <= 0:
            return 1.0
        if not mask_a.any() and not mask_b.any():
            return 1.0
        try:
            distance = hausdorff_distance(mask_a.astype(bool), mask_b.astype(bool))
        except ValueError:
            return 0.0
        normalized = min(distance / diagonal, 1.0)
        return 1.0 - normalized

    def _match_masks(
        self,
        before_masks: List[MaskItem],
        after_masks: List[MaskItem],
        diagonal: float,
    ) -> Tuple[Dict[str, Tuple[MaskItem, MatchMetrics]], List[str], List[str]]:
        matches: Dict[str, Tuple[MaskItem, MatchMetrics]] = {}
        matched_after_ids: set[str] = set()

        for before in before_masks:
            best_score = -1.0
            best_candidate: Optional[Tuple[MaskItem, MatchMetrics]] = None
            for after in after_masks:
                if after.mask_id in matched_after_ids:
                    continue
                metrics = self._compute_similarity_metrics(before, after, diagonal)
                if metrics.score > best_score:
                    best_score = metrics.score
                    best_candidate = (after, metrics)
            if not best_candidate:
                continue
            after_match, metrics = best_candidate
            if metrics.iou >= self.match_iou_threshold or metrics.score >= self.match_score_threshold:
                matches[before.mask_id] = best_candidate
                matched_after_ids.add(after_match.mask_id)

        unmatched_before = [m.mask_id for m in before_masks if m.mask_id not in matches]
        unmatched_after = [m.mask_id for m in after_masks if m.mask_id not in matched_after_ids]
        return matches, unmatched_before, unmatched_after

    # ------------------------------------------------------------------
    def detect_changes(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        min_area: int = 0,
        histogram_matching: Optional[bool] = None,
        hausdorff_weight: Optional[float] = None,
        grid_points_per_side: Optional[int] = None,
    ) -> Dict[str, List]:
        if hausdorff_weight is not None:
            hausdorff_weight = float(np.clip(hausdorff_weight, 0.0, 1.0))
            self.hausdorff_weight = hausdorff_weight
            self.iou_weight = 1.0 - hausdorff_weight
            self._normalize_match_weights()

        if grid_points_per_side is not None and self.use_fixed_prompt_grid:
            self.update_prompt_grid(int(grid_points_per_side))

        if histogram_matching is None:
            histogram_matching = self.enable_histogram_matching

        before_rgb, after_rgb = self._align_inputs(img_before, img_after, histogram_matching)
        masks_before = self.segment(before_rgb, source="before")
        masks_after = self.segment(after_rgb, source="after")

        if min_area > 0:
            masks_before = [m for m in masks_before if m.area >= min_area]
            masks_after = [m for m in masks_after if m.area >= min_area]
        before_lookup = {mask.mask_id: mask for mask in masks_before}

        diagonal = float(np.sqrt(before_rgb.shape[0] ** 2 + before_rgb.shape[1] ** 2))
        matches, removed_ids, new_ids = self._match_masks(masks_before, masks_after, diagonal)

        removed = [m for m in masks_before if m.mask_id in removed_ids]
        new = [m for m in masks_after if m.mask_id in new_ids]
        modified = []
        unchanged = []

        for before_id, (after_mask, metrics) in matches.items():
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
                        "iou": metrics.iou,
                        "hausdorff": metrics.hausdorff_similarity,
                        "score": metrics.score,
                        "area_delta": area_delta,
                        "centroid_shift": centroid_delta,
                    }
                )
            else:
                unchanged.append((before_mask, after_mask, metrics))

        return {
            "masks_before": masks_before,
            "masks_after": masks_after,
            "new": new,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
            "before_image": before_rgb,
            "after_image": after_rgb,
            "match_weights": {"iou": self.iou_weight, "hausdorff": self.hausdorff_weight},
        }

    # ------------------------------------------------------------------
    def _align_inputs(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        histogram_matching: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        before = self._ensure_rgb_uint8(img_before)
        after = self._ensure_rgb_uint8(img_after)
        if before.shape[:2] != after.shape[:2]:
            after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_LINEAR)
        if histogram_matching:
            after = self._match_histograms(after, before)
        return before, after

    @staticmethod
    def _match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        matched = match_histograms(image, reference, channel_axis=2)
        return np.clip(matched, 0, 255).astype(np.uint8)

    def _centroid_shift(self, before: MaskItem, after: MaskItem, image_shape: Tuple[int, int]) -> float:
        before_centroid = np.array(before.centroid)
        after_centroid = np.array(after.centroid)
        height, width = image_shape
        diagonal = np.sqrt(height**2 + width**2)
        if diagonal == 0:
            return 0.0
        return float(np.linalg.norm(after_centroid - before_centroid) / diagonal)
