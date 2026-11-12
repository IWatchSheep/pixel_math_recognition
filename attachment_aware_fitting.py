"""Stage 1G-B: progressive template fitting with attachment-loss penalties."""

from __future__ import annotations

import csv
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable

from PIL import Image

from connected_sections import Point
from growth_experiment_recorder import RecordingCancelled, pictures_directory
from progressive_anchored_fitting import (
    ProgressiveAnchoredFitConfig,
    Transform,
    _fit_score,
    _make_overlay,
    _render_template_mask,
    _select_transform,
    _template_mask,
    _within_cumulative_limits,
    attachment_changes,
    parse_progressive_anchored_fit_config,
)
from threshold_growth import (
    GrowthAnalysisCancelled,
    ThresholdGrowthHistory,
    analyze_threshold_growth,
)

ProgressCallback = Callable[[int, int, str], None]
_WEIGHT = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z")
EXPERIMENT_NAME = "stage_1g_b_attachment_aware"


@dataclass(frozen=True)
class AttachmentAwareFitConfig:
    base_config: ProgressiveAnchoredFitConfig
    attachment_loss_weights: tuple[float, ...]


@dataclass(frozen=True)
class AttachmentAwareFrame:
    digit: str
    attachment_loss_weight: float
    threshold: int
    offset_x: int
    offset_y: int
    rotation_degrees: int
    width_scale: float
    height_scale: float
    transform_change: Transform | None
    transform_bound_hits: tuple[str, ...]
    iou: float
    objective: float
    attachment_loss_fraction: float
    previous_matched_pixels: tuple[Point, ...]
    matched_pixels: tuple[Point, ...]
    retained_attachments: tuple[Point, ...]
    gained_attachments: tuple[Point, ...]
    lost_attachments: tuple[Point, ...]
    retained_fraction: float | None
    cumulative_gained_count: int
    cumulative_lost_count: int
    cumulative_gained_attachments: tuple[Point, ...]
    cumulative_lost_attachments: tuple[Point, ...]
    unmatched_observed_pixels: tuple[Point, ...]
    unsupported_template_pixels: tuple[Point, ...]
    newly_selected_pixels: tuple[Point, ...]
    newly_connected_pixels: tuple[Point, ...]
    overlay: Image.Image

    @property
    def transform(self) -> Transform:
        return (
            self.offset_x,
            self.offset_y,
            self.rotation_degrees,
            self.width_scale,
            self.height_scale,
        )

    @property
    def at_transform_bound(self) -> bool:
        return bool(self.transform_bound_hits)


@dataclass(frozen=True)
class AttachmentAwareFitResult:
    config: AttachmentAwareFitConfig
    image_size: tuple[int, int]
    growth_history: ThresholdGrowthHistory
    frames: tuple[AttachmentAwareFrame, ...]

    def frames_for(
            self, digit: str, attachment_loss_weight: float
    ) -> tuple[AttachmentAwareFrame, ...]:
        return tuple(
            frame for frame in self.frames
            if frame.digit == digit
            and frame.attachment_loss_weight == attachment_loss_weight
        )

    @property
    def display_overlay(self) -> Image.Image:
        digit = self.config.base_config.candidate_digits[0]
        return self.frames_for(digit, 0.0)[-1].overlay


@dataclass(frozen=True)
class AttachmentAwareRecordingResult:
    experiment: AttachmentAwareFitResult
    output_directory: Path
    overlay_count: int


def parse_attachment_aware_fit_config(
        text: str,
        image_size: tuple[int, int],
) -> AttachmentAwareFitConfig:
    weight_values: str | None = None
    base_lines: list[str] = []
    experiment_seen = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("attachment_loss_weights="):
            if weight_values is not None:
                raise ValueError(
                    f"Line {line_number}: duplicate key 'attachment_loss_weights'."
                )
            weight_values = stripped.partition("=")[2].strip()
            if not weight_values:
                raise ValueError(
                    f"Line {line_number}: key and value must be non-empty."
                )
            continue
        if stripped.startswith("experiment="):
            if experiment_seen:
                raise ValueError(f"Line {line_number}: duplicate key 'experiment'.")
            experiment_seen = True
            if stripped.partition("=")[2].strip() != "progressive_anchored_fit_b":
                raise ValueError("'experiment' must be 'progressive_anchored_fit_b'.")
            base_lines.append("experiment=progressive_anchored_fit")
            continue
        base_lines.append(line)
    if weight_values is None:
        raise ValueError("Missing required keys: attachment_loss_weights.")
    if not experiment_seen:
        raise ValueError("Missing required keys: experiment.")

    weights: list[float] = []
    for value in weight_values.split(","):
        value = value.strip()
        if not _WEIGHT.fullmatch(value):
            raise ValueError(
                "'attachment_loss_weights' must be comma-separated finite "
                "non-negative numbers."
            )
        weight = float(value)
        if not math.isfinite(weight):
            raise ValueError(
                "'attachment_loss_weights' must contain finite values."
            )
        if weight < 0:
            raise ValueError(
                "'attachment_loss_weights' must contain non-negative values."
            )
        if weight in weights:
            raise ValueError("'attachment_loss_weights' values must be unique.")
        weights.append(weight)
    if 0.0 not in weights:
        raise ValueError(
            "'attachment_loss_weights' must include 0 for the Stage 1G baseline."
        )

    base_config = parse_progressive_anchored_fit_config(
        "\n".join(base_lines), image_size
    )
    if base_config.experiment_name != EXPERIMENT_NAME:
        raise ValueError(
            f"'experiment_name' must be '{EXPERIMENT_NAME}' so Stage 1G-B "
            "recordings cannot overwrite Stage 1G output."
        )
    return AttachmentAwareFitConfig(base_config, tuple(weights))


def _to_original_coordinates(
        points: frozenset[Point] | tuple[Point, ...],
        roi_left: int,
        roi_top: int,
) -> frozenset[Point]:
    return frozenset((x + roi_left, y + roi_top) for x, y in points)


def _attachment_loss_fraction(
        previous: frozenset[Point],
        current: frozenset[Point],
) -> float:
    return len(previous - current) / max(1, len(previous))


def _select_attachment_aware_transform(
        observed: frozenset[Point],
        template: Image.Image,
        roi_size: tuple[int, int],
        current: Transform,
        initial: Transform,
        config: ProgressiveAnchoredFitConfig,
        previous_matched_pixels: frozenset[Point],
        attachment_loss_weight: float,
) -> Transform:
    if not previous_matched_pixels or attachment_loss_weight == 0:
        return _select_transform(
            observed, template, roi_size, current, initial, config
        )

    selected = current
    dimensions = (
        (0, (-config.step_translation_pixels, 0, config.step_translation_pixels)),
        (1, (-config.step_translation_pixels, 0, config.step_translation_pixels)),
        (2, (-config.step_rotation_degrees, 0, config.step_rotation_degrees)),
        (3, (-config.step_scale, 0.0, config.step_scale)),
        (4, (-config.step_scale, 0.0, config.step_scale)),
    )
    roi_left, roi_top, _roi_right, _roi_bottom = config.roi
    for dimension, deltas in dimensions:
        best_rank: tuple[float, float, int, int, int, int, float, float] | None = None
        best_transform = selected
        for delta in deltas:
            values = list(selected)
            values[dimension] += delta
            if dimension >= 3:
                values[dimension] = round(float(values[dimension]), 6)
            candidate: Transform = (
                int(values[0]), int(values[1]), int(values[2]),
                float(values[3]), float(values[4]),
            )
            if not _within_cumulative_limits(candidate, initial, config):
                continue
            candidate_pixels = _render_template_mask(
                template, roi_size, candidate
            )
            base_objective, iou, matched, _unmatched, _unsupported = _fit_score(
                observed, candidate_pixels, candidate, initial
            )
            matched_original = _to_original_coordinates(
                matched, roi_left, roi_top
            )
            loss_fraction = _attachment_loss_fraction(
                previous_matched_pixels, matched_original
            )
            objective = base_objective - attachment_loss_weight * loss_fraction
            step_distance = sum(
                (abs(float(a) - float(b)) for a, b in zip(candidate, selected)),
                0.0,
            )
            rank = (
                -objective,
                -iou,
                int(step_distance * 1_000_000),
                candidate[1],
                candidate[0],
                candidate[2],
                candidate[3],
                candidate[4],
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_transform = candidate
        selected = best_transform
    return selected


def _transform_bound_hits(
        transform: Transform,
        initial: Transform,
        config: ProgressiveAnchoredFitConfig,
        transform_change: Transform | None,
) -> tuple[str, ...]:
    names = ("offset_x", "offset_y", "rotation", "width_scale", "height_scale")
    cumulative_limits = (
        config.cumulative_translation_pixels,
        config.cumulative_translation_pixels,
        config.cumulative_rotation_degrees,
        config.cumulative_scale,
        config.cumulative_scale,
    )
    step_limits = (
        config.step_translation_pixels,
        config.step_translation_pixels,
        config.step_rotation_degrees,
        config.step_scale,
        config.step_scale,
    )
    hits: list[str] = []
    for index, name in enumerate(names):
        delta = float(transform[index]) - float(initial[index])
        limit = float(cumulative_limits[index])
        if math.isclose(delta, -limit, abs_tol=1e-9):
            hits.append(f"{name}_cumulative_min")
        if math.isclose(delta, limit, abs_tol=1e-9):
            hits.append(f"{name}_cumulative_max")
        if (
                transform_change is not None
                and math.isclose(
                    abs(float(transform_change[index])),
                    float(step_limits[index]),
                    abs_tol=1e-9,
                )
        ):
            hits.append(f"{name}_step")
    return tuple(hits)


def run_attachment_aware_fit(
        config: AttachmentAwareFitConfig,
        rgb_image: Image.Image,
        grayscale_image: Image.Image,
        *,
        cancel_event: Event | None = None,
        progress_callback: ProgressCallback | None = None,
) -> AttachmentAwareFitResult:
    if rgb_image.mode != "RGB" or grayscale_image.mode != "L":
        raise ValueError("Stage 1G-B requires RGB and grayscale source images.")
    if rgb_image.size != grayscale_image.size:
        raise ValueError("RGB and grayscale image dimensions must match.")
    base = config.base_config
    cancellation = cancel_event or Event()
    try:
        growth = analyze_threshold_growth(
            grayscale_image,
            base.roi,
            base.seed,
            base.start_threshold,
            base.maximum_threshold,
            base.threshold_increment,
            base.background_limit_percentage,
            cancel_event=cancellation,
        )
    except GrowthAnalysisCancelled as error:
        raise RecordingCancelled("Stage 1G-B analysis was canceled.") from error

    roi_size = (base.roi[2] - base.roi[0], base.roi[3] - base.roi[1])
    initial: Transform = (
        base.initial_offset_x,
        base.initial_offset_y,
        base.initial_rotation_degrees,
        base.initial_width_scale,
        base.initial_height_scale,
    )
    previous_components: frozenset[Point] = frozenset()
    previous_selected: frozenset[Point] = frozenset()
    previous_attachments = {
        (digit, weight): frozenset()
        for digit in base.candidate_digits
        for weight in config.attachment_loss_weights
    }
    transforms = {
        (digit, weight): initial
        for digit in base.candidate_digits
        for weight in config.attachment_loss_weights
    }
    cumulative_gained = {
        (digit, weight): 0
        for digit in base.candidate_digits
        for weight in config.attachment_loss_weights
    }
    cumulative_lost = dict(cumulative_gained)
    cumulative_gained_points = {
        key: set() for key in cumulative_gained
    }
    cumulative_lost_points = {
        key: set() for key in cumulative_gained
    }
    total = max(
        1,
        len(growth.frames)
        * len(base.candidate_digits)
        * len(config.attachment_loss_weights),
    )
    completed = 0
    results: list[AttachmentAwareFrame] = []
    roi_left, roi_top, _roi_right, _roi_bottom = base.roi

    for frame_index, growth_frame in enumerate(growth.frames):
        if cancellation.is_set():
            raise RecordingCancelled("Stage 1G-B analysis was canceled.")
        tracked = frozenset(growth_frame.component_pixels)
        selected_pixels = []
        for y in range(base.roi[1], base.roi[3]):
            for x in range(base.roi[0], base.roi[2]):
                brightness = grayscale_image.getpixel((x, y))
                if not isinstance(brightness, int):
                    raise ValueError(
                        "Stage 1G-B grayscale image must contain scalar pixels."
                    )
                if brightness <= growth_frame.threshold:
                    selected_pixels.append((x, y))
        selected = frozenset(selected_pixels)
        newly_selected = selected - previous_selected
        newly_connected = frozenset(tracked) - previous_components - newly_selected
        tracked_roi = frozenset(
            (x - roi_left, y - roi_top) for x, y in tracked
        )

        for digit in base.candidate_digits:
            template = _template_mask(digit)
            for weight in config.attachment_loss_weights:
                if cancellation.is_set():
                    raise RecordingCancelled("Stage 1G-B analysis was canceled.")
                key = (digit, weight)
                previous_matches = previous_attachments[key]
                previous_transform = transforms[key]
                transform = previous_transform
                if frame_index > 0:
                    transform = _select_attachment_aware_transform(
                        tracked_roi,
                        template,
                        roi_size,
                        previous_transform,
                        initial,
                        base,
                        previous_matches,
                        weight,
                    )
                transforms[key] = transform
                template_pixels = _render_template_mask(
                    template, roi_size, transform
                )
                base_objective, iou, matched_local, unmatched_local, unsupported_local = (
                    _fit_score(tracked_roi, template_pixels, transform, initial)
                )
                matched = _to_original_coordinates(
                    matched_local, roi_left, roi_top
                )
                unmatched = _to_original_coordinates(
                    unmatched_local, roi_left, roi_top
                )
                unsupported = _to_original_coordinates(
                    unsupported_local, roi_left, roi_top
                )
                retained, gained, lost = attachment_changes(
                    previous_matches, matched
                )
                loss_fraction = _attachment_loss_fraction(
                    previous_matches, matched
                )
                objective = base_objective - weight * loss_fraction
                cumulative_gained[key] += len(gained)
                cumulative_lost[key] += len(lost)
                cumulative_gained_points[key].update(gained)
                cumulative_lost_points[key].update(lost)
                transform_change: Transform | None = None
                if frame_index > 0:
                    transform_change = (
                        transform[0] - previous_transform[0],
                        transform[1] - previous_transform[1],
                        transform[2] - previous_transform[2],
                        round(transform[3] - previous_transform[3], 6),
                        round(transform[4] - previous_transform[4], 6),
                    )
                overlay = _make_overlay(
                    rgb_image,
                    base.roi,
                    tracked,
                    matched,
                    template_pixels,
                )
                results.append(
                    AttachmentAwareFrame(
                        digit=digit,
                        attachment_loss_weight=weight,
                        threshold=growth_frame.threshold,
                        offset_x=transform[0],
                        offset_y=transform[1],
                        rotation_degrees=transform[2],
                        width_scale=transform[3],
                        height_scale=transform[4],
                        transform_change=transform_change,
                        transform_bound_hits=_transform_bound_hits(
                            transform, initial, base, transform_change
                        ),
                        iou=iou,
                        objective=objective,
                        attachment_loss_fraction=loss_fraction,
                        previous_matched_pixels=tuple(sorted(previous_matches)),
                        matched_pixels=tuple(sorted(matched)),
                        retained_attachments=tuple(sorted(retained)),
                        gained_attachments=tuple(sorted(gained)),
                        lost_attachments=tuple(sorted(lost)),
                        retained_fraction=(
                            len(retained) / len(previous_matches)
                            if previous_matches else None
                        ),
                        cumulative_gained_count=cumulative_gained[key],
                        cumulative_lost_count=cumulative_lost[key],
                        cumulative_gained_attachments=tuple(
                            sorted(cumulative_gained_points[key])
                        ),
                        cumulative_lost_attachments=tuple(
                            sorted(cumulative_lost_points[key])
                        ),
                        unmatched_observed_pixels=tuple(sorted(unmatched)),
                        unsupported_template_pixels=tuple(sorted(unsupported)),
                        newly_selected_pixels=tuple(sorted(newly_selected)),
                        newly_connected_pixels=tuple(sorted(newly_connected)),
                        overlay=overlay,
                    )
                )
                previous_attachments[key] = matched
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total,
                        f"Fitting template {digit} at {growth_frame.threshold} "
                        f"(attachment weight {weight:g})",
                    )
        previous_components = frozenset(tracked)
        previous_selected = selected

    return AttachmentAwareFitResult(
        config, rgb_image.size, growth, tuple(results)
    )


def _coordinate_text(points: tuple[Point, ...]) -> str:
    return ";".join(f"{x},{y}" for x, y in points)


def _weight_text(weight: float) -> str:
    return format(weight, ".12g")


def _settings_text(config: AttachmentAwareFitConfig) -> str:
    base_text = "\n".join(
        line for line in (
            f"experiment=progressive_anchored_fit_b",
            f"experiment_name={EXPERIMENT_NAME}",
            f"roi_left={config.base_config.roi[0]}",
            f"roi_top={config.base_config.roi[1]}",
            f"roi_right={config.base_config.roi[2]}",
            f"roi_bottom={config.base_config.roi[3]}",
            f"seed_x={config.base_config.seed[0]}",
            f"seed_y={config.base_config.seed[1]}",
            f"brightness_threshold={config.base_config.start_threshold}",
            f"maximum_threshold={config.base_config.maximum_threshold}",
            f"threshold_increment={config.base_config.threshold_increment}",
            f"background_limit_percent={config.base_config.background_limit_percentage:g}",
            f"candidate_digits={','.join(config.base_config.candidate_digits)}",
            f"initial_offset_x={config.base_config.initial_offset_x}",
            f"initial_offset_y={config.base_config.initial_offset_y}",
            f"initial_rotation_degrees={config.base_config.initial_rotation_degrees}",
            f"initial_width_scale={config.base_config.initial_width_scale:g}",
            f"initial_height_scale={config.base_config.initial_height_scale:g}",
            f"step_translation_pixels={config.base_config.step_translation_pixels}",
            f"cumulative_translation_pixels={config.base_config.cumulative_translation_pixels}",
            f"step_rotation_degrees={config.base_config.step_rotation_degrees}",
            f"cumulative_rotation_degrees={config.base_config.cumulative_rotation_degrees}",
            f"step_scale={config.base_config.step_scale:g}",
            f"cumulative_scale={config.base_config.cumulative_scale:g}",
            f"attachment_loss_weights={','.join(_weight_text(w) for w in config.attachment_loss_weights)}",
            f"auto_record={str(config.base_config.auto_record).lower()}",
        )
        if line
    )
    return base_text + "\n"


def _unique_run_directory(output_root: Path) -> Path:
    parent = output_root / EXPERIMENT_NAME
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(1000):
        run_directory = parent / f"{timestamp}_{suffix:03d}"
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        return run_directory
    raise FileExistsError("Could not create a unique Stage 1G-B output directory.")


def record_attachment_aware_fit(
        config: AttachmentAwareFitConfig,
        rgb_image: Image.Image,
        grayscale_image: Image.Image,
        output_root: Path | None = None,
        *,
        cancel_event: Event | None = None,
        progress_callback: ProgressCallback | None = None,
) -> AttachmentAwareRecordingResult:
    if not config.base_config.auto_record:
        raise ValueError("Stage 1G-B recording requires auto_record=true.")
    cancellation = cancel_event or Event()
    result = run_attachment_aware_fit(
        config,
        rgb_image,
        grayscale_image,
        cancel_event=cancellation,
        progress_callback=progress_callback,
    )
    root = output_root or (pictures_directory() / "PixelMathExperiments")
    run_directory = _unique_run_directory(root)
    overlays_directory = run_directory / "overlays"
    overlays_directory.mkdir()
    try:
        history_fields = (
            "digit", "attachment_loss_weight", "threshold", "offset_x", "offset_y",
            "rotation_degrees", "width_scale", "height_scale", "transform_change",
            "transform_bound_hits", "at_transform_bound", "iou",
            "existing_transform_penalty",
            "attachment_loss_fraction", "attachment_penalty", "objective",
            "matched_count", "previous_match_count", "retained_count", "gained_count",
            "lost_count", "retained_fraction", "cumulative_gained_count",
            "cumulative_lost_count", "cumulative_gained_attachments",
            "cumulative_lost_attachments", "unmatched_observed_count",
            "unsupported_template_count", "previous_matched_pixels", "matched_pixels",
            "retained_attachments", "gained_attachments", "lost_attachments",
            "unmatched_observed_pixels", "unsupported_template_pixels",
            "newly_selected_count", "newly_selected_pixels",
            "newly_connected_count", "newly_connected_pixels",
        )
        history_path = run_directory / "fitting_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history_fields)
            writer.writeheader()
            for frame in result.frames:
                if cancellation.is_set():
                    raise RecordingCancelled("Stage 1G-B recording was canceled.")
                transform_penalty = (
                    0.002 * (
                        abs(frame.offset_x - result.config.base_config.initial_offset_x)
                        + abs(frame.offset_y - result.config.base_config.initial_offset_y)
                    )
                    + 0.0005 * abs(
                        frame.rotation_degrees
                        - result.config.base_config.initial_rotation_degrees
                    )
                    + 0.05 * (
                        abs(frame.width_scale - result.config.base_config.initial_width_scale)
                        + abs(frame.height_scale - result.config.base_config.initial_height_scale)
                    )
                )
                attachment_penalty = (
                    frame.attachment_loss_weight * frame.attachment_loss_fraction
                )
                transform_change = (
                    ""
                    if frame.transform_change is None
                    else ",".join(format(value, ".6g") for value in frame.transform_change)
                )
                writer.writerow({
                    "digit": frame.digit,
                    "attachment_loss_weight": _weight_text(frame.attachment_loss_weight),
                    "threshold": frame.threshold,
                    "offset_x": frame.offset_x,
                    "offset_y": frame.offset_y,
                    "rotation_degrees": frame.rotation_degrees,
                    "width_scale": frame.width_scale,
                    "height_scale": frame.height_scale,
                    "transform_change": transform_change,
                    "transform_bound_hits": ";".join(frame.transform_bound_hits),
                    "at_transform_bound": frame.at_transform_bound,
                    "iou": f"{frame.iou:.8f}",
                    "existing_transform_penalty": f"{transform_penalty:.8f}",
                    "attachment_loss_fraction": f"{frame.attachment_loss_fraction:.8f}",
                    "attachment_penalty": f"{attachment_penalty:.8f}",
                    "objective": f"{frame.objective:.8f}",
                    "matched_count": len(frame.matched_pixels),
                    "previous_match_count": len(frame.previous_matched_pixels),
                    "retained_count": len(frame.retained_attachments),
                    "gained_count": len(frame.gained_attachments),
                    "lost_count": len(frame.lost_attachments),
                    "retained_fraction": (
                        "" if frame.retained_fraction is None
                        else f"{frame.retained_fraction:.8f}"
                    ),
                    "cumulative_gained_count": frame.cumulative_gained_count,
                    "cumulative_lost_count": frame.cumulative_lost_count,
                    "cumulative_gained_attachments": _coordinate_text(
                        frame.cumulative_gained_attachments
                    ),
                    "cumulative_lost_attachments": _coordinate_text(
                        frame.cumulative_lost_attachments
                    ),
                    "unmatched_observed_count": len(frame.unmatched_observed_pixels),
                    "unsupported_template_count": len(frame.unsupported_template_pixels),
                    "previous_matched_pixels": _coordinate_text(frame.previous_matched_pixels),
                    "matched_pixels": _coordinate_text(frame.matched_pixels),
                    "retained_attachments": _coordinate_text(frame.retained_attachments),
                    "gained_attachments": _coordinate_text(frame.gained_attachments),
                    "lost_attachments": _coordinate_text(frame.lost_attachments),
                    "unmatched_observed_pixels": _coordinate_text(frame.unmatched_observed_pixels),
                    "unsupported_template_pixels": _coordinate_text(frame.unsupported_template_pixels),
                    "newly_selected_count": len(frame.newly_selected_pixels),
                    "newly_selected_pixels": _coordinate_text(frame.newly_selected_pixels),
                    "newly_connected_count": len(frame.newly_connected_pixels),
                    "newly_connected_pixels": _coordinate_text(frame.newly_connected_pixels),
                })
                filename = (
                    f"digit_{frame.digit}_weight_{_weight_text(frame.attachment_loss_weight)}"
                    f"_threshold_{frame.threshold:03d}.png"
                )
                frame.overlay.save(overlays_directory / filename, format="PNG")

        growth_path = run_directory / "growth_measurements.csv"
        with growth_path.open("w", newline="", encoding="utf-8") as file:
            fields = (
                "threshold", "newly_selected_count", "newly_selected_pixels",
                "newly_connected_count", "newly_connected_pixels",
                "tracked_component_count", "selected_roi_count",
                "absorbed_start_sections",
            )
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for growth_frame in result.growth_history.frames:
                sample = next(
                    frame for frame in result.frames
                    if frame.threshold == growth_frame.threshold
                    and frame.digit == result.config.base_config.candidate_digits[0]
                    and frame.attachment_loss_weight == 0.0
                )
                writer.writerow({
                    "threshold": growth_frame.threshold,
                    "newly_selected_count": len(sample.newly_selected_pixels),
                    "newly_selected_pixels": _coordinate_text(sample.newly_selected_pixels),
                    "newly_connected_count": len(sample.newly_connected_pixels),
                    "newly_connected_pixels": _coordinate_text(sample.newly_connected_pixels),
                    "tracked_component_count": growth_frame.pixel_count,
                    "selected_roi_count": growth_frame.selected_pixel_count,
                    "absorbed_start_sections": growth_frame.absorbed_start_sections,
                })

        comparison_path = run_directory / "weight_comparisons.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as file:
            fields = (
                "digit", "threshold", "attachment_loss_weight", "baseline_iou",
                "current_iou", "iou_delta_from_baseline", "baseline_objective",
                "current_objective", "matched_count", "retained_count",
                "gained_count", "lost_count", "retained_fraction",
                "cumulative_gained_count", "cumulative_lost_count",
                "offset_x", "offset_y", "rotation_degrees", "width_scale",
                "height_scale",
            )
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for digit in result.config.base_config.candidate_digits:
                baseline = {
                    frame.threshold: frame
                    for frame in result.frames_for(digit, 0.0)
                }
                for weight in result.config.attachment_loss_weights:
                    for frame in result.frames_for(digit, weight):
                        baseline_frame = baseline[frame.threshold]
                        writer.writerow({
                            "digit": digit,
                            "threshold": frame.threshold,
                            "attachment_loss_weight": _weight_text(weight),
                            "baseline_iou": f"{baseline_frame.iou:.8f}",
                            "current_iou": f"{frame.iou:.8f}",
                            "iou_delta_from_baseline": (
                                f"{frame.iou - baseline_frame.iou:.8f}"
                            ),
                            "baseline_objective": f"{baseline_frame.objective:.8f}",
                            "current_objective": f"{frame.objective:.8f}",
                            "matched_count": len(frame.matched_pixels),
                            "retained_count": len(frame.retained_attachments),
                            "gained_count": len(frame.gained_attachments),
                            "lost_count": len(frame.lost_attachments),
                            "retained_fraction": (
                                "" if frame.retained_fraction is None
                                else f"{frame.retained_fraction:.8f}"
                            ),
                            "cumulative_gained_count": frame.cumulative_gained_count,
                            "cumulative_lost_count": frame.cumulative_lost_count,
                            "offset_x": frame.offset_x,
                            "offset_y": frame.offset_y,
                            "rotation_degrees": frame.rotation_degrees,
                            "width_scale": frame.width_scale,
                            "height_scale": frame.height_scale,
                        })

        settings_path = run_directory / "experiment_settings.txt"
        settings_path.write_text(_settings_text(config), encoding="utf-8")
        zip_path = run_directory / "results.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            files = sorted(
                path for path in run_directory.rglob("*")
                if path.is_file() and path != zip_path
            )
            for index, path in enumerate(files):
                if cancellation.is_set():
                    raise RecordingCancelled("Stage 1G-B packaging was canceled.")
                archive.write(path, path.relative_to(run_directory))
                if progress_callback is not None:
                    progress_callback(
                        index + 1, len(files), f"Packaged {path.name}"
                    )
        return AttachmentAwareRecordingResult(
            result, run_directory, len(result.frames)
        )
    except Exception:
        shutil.rmtree(run_directory, ignore_errors=False)
        raise
