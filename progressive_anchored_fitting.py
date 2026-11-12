"""Progressive, bounded template attachment experiments (Stage 1G)."""

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

from PIL import Image, ImageDraw

from connected_sections import Point, Roi
from growth_experiment_recorder import RecordingCancelled, pictures_directory
from template_matching import DIGIT_TEMPLATE_ROWS
from threshold_growth import (
    GrowthAnalysisCancelled,
    ThresholdGrowthHistory,
    analyze_threshold_growth,
)

ProgressCallback = Callable[[int, int, str], None]
Transform = tuple[int, int, int, float, float]
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")
_UNSIGNED_INTEGER = re.compile(r"\d+\Z")
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z")
_CANDIDATE_DIGITS = frozenset(("0", "6", "8", "9"))
_OVERLAY_COLORS = {
    "matched": (0, 210, 80),
    "unmatched": (255, 96, 48),
    "unsupported": (64, 128, 255),
    "boundary": (0, 220, 255),
}

_REQUIRED_KEYS = frozenset(
    {
        "experiment",
        "roi_left",
        "roi_top",
        "roi_right",
        "roi_bottom",
        "seed_x",
        "seed_y",
        "brightness_threshold",
        "maximum_threshold",
        "threshold_increment",
        "background_limit_percent",
        "candidate_digits",
        "initial_offset_x",
        "initial_offset_y",
        "initial_rotation_degrees",
        "initial_width_scale",
        "initial_height_scale",
        "step_translation_pixels",
        "cumulative_translation_pixels",
        "step_rotation_degrees",
        "cumulative_rotation_degrees",
        "step_scale",
        "cumulative_scale",
        "auto_record",
        "experiment_name",
    }
)
_INTEGER_KEYS = (
    "roi_left",
    "roi_top",
    "roi_right",
    "roi_bottom",
    "seed_x",
    "seed_y",
    "brightness_threshold",
    "maximum_threshold",
    "threshold_increment",
    "initial_offset_x",
    "initial_offset_y",
    "initial_rotation_degrees",
    "step_translation_pixels",
    "cumulative_translation_pixels",
    "step_rotation_degrees",
    "cumulative_rotation_degrees",
)
_FLOAT_KEYS = (
    "initial_width_scale",
    "initial_height_scale",
    "step_scale",
    "cumulative_scale",
)


@dataclass(frozen=True)
class ProgressiveAnchoredFitConfig:
    experiment: str
    roi: Roi
    seed: Point
    start_threshold: int
    maximum_threshold: int
    threshold_increment: int
    background_limit_percentage: float
    candidate_digits: tuple[str, ...]
    initial_offset_x: int
    initial_offset_y: int
    initial_rotation_degrees: int
    initial_width_scale: float
    initial_height_scale: float
    step_translation_pixels: int
    cumulative_translation_pixels: int
    step_rotation_degrees: int
    cumulative_rotation_degrees: int
    step_scale: float
    cumulative_scale: float
    auto_record: bool
    experiment_name: str


@dataclass(frozen=True)
class FittingFrame:
    digit: str
    method: str
    threshold: int
    offset_x: int
    offset_y: int
    rotation_degrees: int
    width_scale: float
    height_scale: float
    iou: float
    objective: float
    matched_pixels: tuple[Point, ...]
    retained_attachments: tuple[Point, ...]
    gained_attachments: tuple[Point, ...]
    lost_attachments: tuple[Point, ...]
    unmatched_observed_pixels: tuple[Point, ...]
    unsupported_template_pixels: tuple[Point, ...]
    newly_selected_pixels: tuple[Point, ...]
    newly_connected_pixels: tuple[Point, ...]
    overlay: Image.Image


@dataclass(frozen=True)
class ProgressiveAnchoredFitResult:
    config: ProgressiveAnchoredFitConfig
    image_size: tuple[int, int]
    growth_history: ThresholdGrowthHistory
    frames: tuple[FittingFrame, ...]

    def frames_for(self, digit: str, method: str) -> tuple[FittingFrame, ...]:
        return tuple(
            frame for frame in self.frames
            if frame.digit == digit and frame.method == method
        )

    @property
    def display_overlay(self) -> Image.Image:
        digit = self.config.candidate_digits[0]
        frames = self.frames_for(digit, "progressive")
        return frames[-1].overlay


@dataclass(frozen=True)
class ProgressiveAnchoredRecordingResult:
    experiment: ProgressiveAnchoredFitResult
    output_directory: Path
    overlay_count: int


def parse_progressive_anchored_fit_config(
        text: str,
        image_size: tuple[int, int],
) -> ProgressiveAnchoredFitConfig:
    values: dict[str, str] = {}
    allowed = _REQUIRED_KEYS
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"Line {line_number}: expected key=value.")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or not value:
            raise ValueError(f"Line {line_number}: key and value must be non-empty.")
        if key not in allowed:
            raise ValueError(f"Line {line_number}: unknown key '{key}'.")
        if key in values:
            raise ValueError(f"Line {line_number}: duplicate key '{key}'.")
        values[key] = value
    missing = sorted(_REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}.")
    if values["experiment"] != "progressive_anchored_fit":
        raise ValueError("'experiment' must be 'progressive_anchored_fit'.")

    integers: dict[str, int] = {}
    for key in _INTEGER_KEYS:
        pattern = _INTEGER if key in {"initial_offset_x", "initial_offset_y"} else _UNSIGNED_INTEGER
        if not pattern.fullmatch(values[key]):
            raise ValueError(f"'{key}' must be an integer.")
        integers[key] = int(values[key])
    numbers: dict[str, float] = {}
    for key in _FLOAT_KEYS:
        if not _NUMBER.fullmatch(values[key]):
            raise ValueError(f"'{key}' must be a finite number.")
        number = float(values[key])
        if not math.isfinite(number):
            raise ValueError(f"'{key}' must be a finite number.")
        numbers[key] = number
    if values["auto_record"] not in {"true", "false"}:
        raise ValueError("'auto_record' must be 'true' or 'false'.")
    experiment_name = values["experiment_name"]
    if not _SAFE_NAME.fullmatch(experiment_name):
        raise ValueError("Unsafe experiment_name; use 1–48 letters, digits, '_' or '-'.")

    digits = tuple(item.strip() for item in values["candidate_digits"].split(","))
    if (
            not digits
            or len(digits) != len(set(digits))
            or any(digit not in _CANDIDATE_DIGITS for digit in digits)
    ):
        raise ValueError("'candidate_digits' must be a unique comma-separated subset of 0,6,8,9.")
    start = integers["brightness_threshold"]
    maximum = integers["maximum_threshold"]
    increment = integers["threshold_increment"]
    if not 0 <= start <= 255 or not start <= maximum <= 255:
        raise ValueError("Thresholds must satisfy 0 <= start <= maximum <= 255.")
    if increment < 1:
        raise ValueError("'threshold_increment' must be positive.")
    occupancy = float(values["background_limit_percent"])
    if not math.isfinite(occupancy) or not 0 < occupancy <= 100:
        raise ValueError("'background_limit_percent' must be greater than 0 and at most 100.")
    for key in ("initial_width_scale", "initial_height_scale"):
        if not 0.5 <= numbers[key] <= 1.5:
            raise ValueError(f"'{key}' must be between 0.5 and 1.5.")
    if not 0.01 <= numbers["step_scale"] <= 0.5:
        raise ValueError("'step_scale' must be between 0.01 and 0.5.")
    if not 0.01 <= numbers["cumulative_scale"] <= 0.5:
        raise ValueError("'cumulative_scale' must be between 0.01 and 0.5.")
    if numbers["step_scale"] > numbers["cumulative_scale"]:
        raise ValueError("'step_scale' cannot exceed 'cumulative_scale'.")
    if integers["step_translation_pixels"] < 1:
        raise ValueError("'step_translation_pixels' must be positive.")
    if integers["cumulative_translation_pixels"] < integers["step_translation_pixels"]:
        raise ValueError("'cumulative_translation_pixels' must be at least the per-step limit.")
    if integers["step_rotation_degrees"] < 1:
        raise ValueError("'step_rotation_degrees' must be positive.")
    if integers["cumulative_rotation_degrees"] < integers["step_rotation_degrees"]:
        raise ValueError("'cumulative_rotation_degrees' must be at least the per-step limit.")

    left, top, right, bottom = (
        integers["roi_left"], integers["roi_top"],
        integers["roi_right"], integers["roi_bottom"],
    )
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"ROI must be a non-empty rectangle inside the loaded {width}×{height} image.")
    seed = integers["seed_x"], integers["seed_y"]
    if not (left <= seed[0] < right and top <= seed[1] < bottom):
        raise ValueError("Seed must be inside the ROI.")
    if not 0 <= integers["initial_rotation_degrees"] <= 359:
        raise ValueError("'initial_rotation_degrees' must be from 0 to 359.")

    return ProgressiveAnchoredFitConfig(
        values["experiment"],
        (left, top, right, bottom),
        seed,
        start,
        maximum,
        increment,
        occupancy,
        digits,
        integers["initial_offset_x"],
        integers["initial_offset_y"],
        integers["initial_rotation_degrees"],
        numbers["initial_width_scale"],
        numbers["initial_height_scale"],
        integers["step_translation_pixels"],
        integers["cumulative_translation_pixels"],
        integers["step_rotation_degrees"],
        integers["cumulative_rotation_degrees"],
        numbers["step_scale"],
        numbers["cumulative_scale"],
        values["auto_record"] == "true",
        experiment_name,
    )


def _template_mask(digit: str) -> Image.Image:
    rows = DIGIT_TEMPLATE_ROWS[digit]
    image = Image.new("L", (len(rows[0]), len(rows)), 255)
    pixels = image.load()
    if pixels is None:
        raise RuntimeError("Could not access the candidate template pixels.")
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value == "1":
                pixels[x, y] = 0
    return image


def _render_template_mask(
        template: Image.Image,
        roi_size: tuple[int, int],
        transform: Transform,
) -> frozenset[Point]:
    offset_x, offset_y, rotation, width_scale, height_scale = transform
    roi_width, roi_height = roi_size
    base_scale = min(roi_width / template.width, roi_height / template.height)
    resized = template.resize(
        (
            max(1, round(template.width * base_scale * width_scale)),
            max(1, round(template.height * base_scale * height_scale)),
        ),
        Image.Resampling.NEAREST,
    )
    rotated = resized.rotate(rotation, resample=Image.Resampling.NEAREST, expand=True)
    origin_x = round((roi_width - rotated.width) / 2 + offset_x)
    origin_y = round((roi_height - rotated.height) / 2 + offset_y)
    local = rotated.load()
    if local is None:
        raise RuntimeError("Could not access the transformed template pixels.")
    return frozenset(
        (x + origin_x, y + origin_y)
        for y in range(rotated.height)
        for x in range(rotated.width)
        if local[x, y] == 0
        and 0 <= x + origin_x < roi_width
        and 0 <= y + origin_y < roi_height
    )


def _fit_score(
        observed: frozenset[Point],
        template_pixels: frozenset[Point],
        transform: Transform,
        initial: Transform,
) -> tuple[float, float, frozenset[Point], frozenset[Point], frozenset[Point]]:
    matched = observed & template_pixels
    unmatched = observed - template_pixels
    unsupported = template_pixels - observed
    union_count = len(matched) + len(unmatched) + len(unsupported)
    iou = len(matched) / union_count if union_count else 0.0
    dx, dy, angle, width_scale, height_scale = transform
    initial_x, initial_y, initial_angle, initial_width, initial_height = initial
    penalty = (
            0.002 * (abs(dx - initial_x) + abs(dy - initial_y))
            + 0.0005 * abs(angle - initial_angle)
            + 0.05 * (abs(width_scale - initial_width) + abs(height_scale - initial_height))
    )
    return iou - penalty, iou, matched, unmatched, unsupported


def attachment_changes(
        previous: frozenset[Point],
        current: frozenset[Point],
) -> tuple[frozenset[Point], frozenset[Point], frozenset[Point]]:
    """Return retained, gained, and lost attachments; a later regain is newly gained."""
    return previous & current, current - previous, previous - current


def _within_cumulative_limits(
        candidate: Transform,
        initial: Transform,
        config: ProgressiveAnchoredFitConfig,
) -> bool:
    x, y, angle, width_scale, height_scale = candidate
    ix, iy, ia, iw, ih = initial
    return (
            abs(x - ix) <= config.cumulative_translation_pixels
            and abs(y - iy) <= config.cumulative_translation_pixels
            and abs(angle - ia) <= config.cumulative_rotation_degrees
            and abs(width_scale - iw) <= config.cumulative_scale + 1e-9
            and abs(height_scale - ih) <= config.cumulative_scale + 1e-9
    )


def _select_transform(
        observed: frozenset[Point],
        template: Image.Image,
        roi_size: tuple[int, int],
        current: Transform,
        initial: Transform,
        config: ProgressiveAnchoredFitConfig,
) -> Transform:
    selected = current
    dimensions = (
        (0, (-config.step_translation_pixels, 0, config.step_translation_pixels)),
        (1, (-config.step_translation_pixels, 0, config.step_translation_pixels)),
        (2, (-config.step_rotation_degrees, 0, config.step_rotation_degrees)),
        (3, (-config.step_scale, 0.0, config.step_scale)),
        (4, (-config.step_scale, 0.0, config.step_scale)),
    )
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
            template_pixels = _render_template_mask(template, roi_size, candidate)
            objective, iou, _matched, _unmatched, _unsupported = _fit_score(
                observed, template_pixels, candidate, initial
            )
            step_distance = sum(abs(float(a) - float(b)) for a, b in zip(candidate, selected))
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


def _make_overlay(
        rgb_image: Image.Image,
        roi: Roi,
        tracked_pixels: frozenset[Point],
        matched_pixels: frozenset[Point],
        template_pixels: frozenset[Point],
) -> Image.Image:
    left, top, right, bottom = roi
    overlay = rgb_image.crop(roi).copy()
    pixels = overlay.load()
    if pixels is None:
        raise RuntimeError("Could not access the overlay pixels.")
    for x, y in tracked_pixels:
        pixels[x - left, y - top] = _OVERLAY_COLORS["unmatched"]
    tracked_local = frozenset((x - left, y - top) for x, y in tracked_pixels)
    for x, y in template_pixels - tracked_local:
        pixels[x, y] = _OVERLAY_COLORS["unsupported"]
    for x, y in matched_pixels:
        pixels[x - left, y - top] = _OVERLAY_COLORS["matched"]
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, overlay.width - 1, overlay.height - 1), outline=_OVERLAY_COLORS["boundary"])
    return overlay


def run_progressive_anchored_fit(
        config: ProgressiveAnchoredFitConfig,
        rgb_image: Image.Image,
        grayscale_image: Image.Image,
        *,
        cancel_event: Event | None = None,
        progress_callback: ProgressCallback | None = None,
) -> ProgressiveAnchoredFitResult:
    if rgb_image.mode != "RGB" or grayscale_image.mode != "L":
        raise ValueError("Stage 1G requires RGB and grayscale source images.")
    if rgb_image.size != grayscale_image.size:
        raise ValueError("RGB and grayscale image dimensions must match.")
    cancellation = cancel_event or Event()
    try:
        growth = analyze_threshold_growth(
            grayscale_image,
            config.roi,
            config.seed,
            config.start_threshold,
            config.maximum_threshold,
            config.threshold_increment,
            config.background_limit_percentage,
            cancel_event=cancellation,
        )
    except GrowthAnalysisCancelled as error:
        raise RecordingCancelled("Stage 1G analysis was canceled.") from error

    roi_size = (config.roi[2] - config.roi[0], config.roi[3] - config.roi[1])
    initial: Transform = (
        config.initial_offset_x,
        config.initial_offset_y,
        config.initial_rotation_degrees,
        config.initial_width_scale,
        config.initial_height_scale,
    )
    previous_components: frozenset[Point] = frozenset()
    previous_selected: frozenset[Point] = frozenset()
    results: list[FittingFrame] = []
    previous_attachments = {
        (digit, method): frozenset()
        for digit in config.candidate_digits
        for method in ("fixed", "progressive")
    }
    progressive_transforms = {
        digit: initial for digit in config.candidate_digits
    }
    total = max(1, len(growth.frames) * len(config.candidate_digits) * 2)
    completed = 0
    for frame_index, growth_frame in enumerate(growth.frames):
        if cancellation.is_set():
            raise RecordingCancelled("Stage 1G analysis was canceled.")
        tracked = frozenset(growth_frame.component_pixels)
        selected_pixels = []
        for y in range(config.roi[1], config.roi[3]):
            for x in range(config.roi[0], config.roi[2]):
                brightness = grayscale_image.getpixel((x, y))
                if not isinstance(brightness, int):
                    raise ValueError("Stage 1G grayscale image must contain scalar pixels.")
                if brightness <= growth_frame.threshold:
                    selected_pixels.append((x, y))
        selected = frozenset(selected_pixels)
        newly_selected = selected - previous_selected
        newly_connected = tracked - previous_components - newly_selected
        roi_left, roi_top, _roi_right, _roi_bottom = config.roi
        tracked_roi = frozenset((x - roi_left, y - roi_top) for x, y in tracked)
        for digit in config.candidate_digits:
            template = _template_mask(digit)
            fixed_state = initial
            progressive_state = progressive_transforms[digit]
            for method in ("fixed", "progressive"):
                if cancellation.is_set():
                    raise RecordingCancelled("Stage 1G analysis was canceled.")
                if method == "progressive":
                    if frame_index > 0:
                        progressive_state = _select_transform(
                            tracked_roi, template, roi_size,
                            progressive_state, initial, config,
                        )
                    progressive_transforms[digit] = progressive_state
                    transform = progressive_state
                else:
                    transform = fixed_state
                template_pixels_roi = _render_template_mask(template, roi_size, transform)
                objective, iou, matched, unmatched, unsupported = _fit_score(
                    tracked_roi, template_pixels_roi, transform, initial
                )
                matched = frozenset((x + roi_left, y + roi_top) for x, y in matched)
                unmatched = frozenset((x + roi_left, y + roi_top) for x, y in unmatched)
                unsupported = frozenset((x + roi_left, y + roi_top) for x, y in unsupported)
                retained, gained, lost = attachment_changes(
                    previous_attachments[(digit, method)], matched
                )
                previous_attachments[(digit, method)] = matched
                overlay = _make_overlay(
                    rgb_image,
                    config.roi,
                    tracked,
                    matched,
                    template_pixels_roi,
                )
                results.append(
                    FittingFrame(
                        digit, method, growth_frame.threshold,
                        transform[0], transform[1], transform[2],
                        transform[3], transform[4], iou, objective,
                        tuple(sorted(matched)), tuple(sorted(retained)),
                        tuple(sorted(gained)), tuple(sorted(lost)),
                        tuple(sorted(unmatched)), tuple(sorted(unsupported)),
                        tuple(sorted(newly_selected)), tuple(sorted(newly_connected)),
                        overlay,
                    )
                )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, f"Fitting template {digit} at {growth_frame.threshold}")
        previous_components = tracked
        previous_selected = selected
    return ProgressiveAnchoredFitResult(
        config, rgb_image.size, growth, tuple(results)
    )


def _coordinate_text(points: tuple[Point, ...]) -> str:
    return "|".join(f"{x},{y}" for x, y in points)


def _settings_text(config: ProgressiveAnchoredFitConfig) -> str:
    fields = (
        ("experiment", config.experiment),
        ("experiment_name", config.experiment_name),
        ("roi_left", config.roi[0]), ("roi_top", config.roi[1]),
        ("roi_right", config.roi[2]), ("roi_bottom", config.roi[3]),
        ("seed_x", config.seed[0]), ("seed_y", config.seed[1]),
        ("brightness_threshold", config.start_threshold),
        ("maximum_threshold", config.maximum_threshold),
        ("threshold_increment", config.threshold_increment),
        ("background_limit_percent", config.background_limit_percentage),
        ("candidate_digits", ",".join(config.candidate_digits)),
        ("initial_offset_x", config.initial_offset_x),
        ("initial_offset_y", config.initial_offset_y),
        ("initial_rotation_degrees", config.initial_rotation_degrees),
        ("initial_width_scale", config.initial_width_scale),
        ("initial_height_scale", config.initial_height_scale),
        ("step_translation_pixels", config.step_translation_pixels),
        ("cumulative_translation_pixels", config.cumulative_translation_pixels),
        ("step_rotation_degrees", config.step_rotation_degrees),
        ("cumulative_rotation_degrees", config.cumulative_rotation_degrees),
        ("step_scale", config.step_scale),
        ("cumulative_scale", config.cumulative_scale),
        ("auto_record", config.auto_record),
    )
    return "".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}\n"
        for key, value in fields
    )


def _write_recording(
        result: ProgressiveAnchoredFitResult,
        output_root: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback | None,
) -> ProgressiveAnchoredRecordingResult:
    if not _SAFE_NAME.fullmatch(result.config.experiment_name):
        raise ValueError("Unsafe experiment name.")
    parent = output_root / result.config.experiment_name
    parent.mkdir(parents=True, exist_ok=True)
    run_directory = parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory.mkdir()
    overlays_directory = run_directory / "overlays"
    overlays_directory.mkdir()
    try:
        history_path = run_directory / "fitting_history.csv"
        history_fields = (
            "digit", "method", "threshold", "offset_x", "offset_y",
            "rotation_degrees", "width_scale", "height_scale", "iou",
            "cumulative_offset_x", "cumulative_offset_y",
            "cumulative_rotation_degrees", "cumulative_width_scale_delta",
            "cumulative_height_scale_delta",
            "objective", "matched_count", "retained_count", "gained_count",
            "lost_count", "unmatched_observed_count", "unsupported_template_count",
            "matched_pixels", "retained_attachments", "gained_attachments",
            "lost_attachments", "unmatched_observed_pixels",
            "unsupported_template_pixels",
        )
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history_fields)
            writer.writeheader()
            for index, frame in enumerate(result.frames):
                if cancel_event.is_set():
                    raise RecordingCancelled("Stage 1G recording was canceled.")
                writer.writerow({
                    "digit": frame.digit,
                    "method": frame.method,
                    "threshold": frame.threshold,
                    "offset_x": frame.offset_x,
                    "offset_y": frame.offset_y,
                    "rotation_degrees": frame.rotation_degrees,
                    "width_scale": frame.width_scale,
                    "height_scale": frame.height_scale,
                    "cumulative_offset_x": (
                            frame.offset_x - result.config.initial_offset_x
                    ),
                    "cumulative_offset_y": (
                            frame.offset_y - result.config.initial_offset_y
                    ),
                    "cumulative_rotation_degrees": (
                            frame.rotation_degrees
                            - result.config.initial_rotation_degrees
                    ),
                    "cumulative_width_scale_delta": (
                        f"{frame.width_scale - result.config.initial_width_scale:.6f}"
                    ),
                    "cumulative_height_scale_delta": (
                        f"{frame.height_scale - result.config.initial_height_scale:.6f}"
                    ),
                    "iou": f"{frame.iou:.8f}",
                    "objective": f"{frame.objective:.8f}",
                    "matched_count": len(frame.matched_pixels),
                    "retained_count": len(frame.retained_attachments),
                    "gained_count": len(frame.gained_attachments),
                    "lost_count": len(frame.lost_attachments),
                    "unmatched_observed_count": len(frame.unmatched_observed_pixels),
                    "unsupported_template_count": len(frame.unsupported_template_pixels),
                    "matched_pixels": _coordinate_text(frame.matched_pixels),
                    "retained_attachments": _coordinate_text(frame.retained_attachments),
                    "gained_attachments": _coordinate_text(frame.gained_attachments),
                    "lost_attachments": _coordinate_text(frame.lost_attachments),
                    "unmatched_observed_pixels": _coordinate_text(frame.unmatched_observed_pixels),
                    "unsupported_template_pixels": _coordinate_text(frame.unsupported_template_pixels),
                })
                filename = f"digit_{frame.digit}_{frame.method}_threshold_{frame.threshold:03d}.png"
                frame.overlay.save(overlays_directory / filename, format="PNG")
                if progress_callback is not None:
                    progress_callback(
                        index + 1, len(result.frames),
                        f"Saved {filename}",
                    )

        growth_path = run_directory / "growth_measurements.csv"
        with growth_path.open("w", newline="", encoding="utf-8") as file:
            fields = (
                "threshold", "newly_selected_count", "newly_selected_pixels",
                "newly_connected_count", "newly_connected_pixels",
                "tracked_component_count", "selected_roi_count", "absorbed_start_sections",
            )
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for frame in result.growth_history.frames:
                # The exact newly selected/connected sets are attached to fit rows;
                # expose the first deterministic candidate/method's duplicate-free
                # growth measurements in this compact threshold-level report.
                sample = next(
                    item for item in result.frames
                    if item.threshold == frame.threshold
                    and item.digit == result.config.candidate_digits[0]
                    and item.method == "fixed"
                )
                writer.writerow({
                    "threshold": frame.threshold,
                    "newly_selected_count": len(sample.newly_selected_pixels),
                    "newly_selected_pixels": _coordinate_text(sample.newly_selected_pixels),
                    "newly_connected_count": len(sample.newly_connected_pixels),
                    "newly_connected_pixels": _coordinate_text(sample.newly_connected_pixels),
                    "tracked_component_count": frame.pixel_count,
                    "selected_roi_count": frame.selected_pixel_count,
                    "absorbed_start_sections": frame.absorbed_start_sections,
                })

        comparison_path = run_directory / "fixed_vs_progressive.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "digit", "threshold", "fixed_iou", "progressive_iou",
                    "fixed_matched_count", "progressive_matched_count",
                    "fixed_retained_count", "progressive_retained_count",
                    "progressive_offset_x", "progressive_offset_y",
                    "progressive_rotation_degrees", "progressive_width_scale",
                    "progressive_height_scale",
                ),
            )
            writer.writeheader()
            for digit in result.config.candidate_digits:
                fixed = result.frames_for(digit, "fixed")
                progressive = result.frames_for(digit, "progressive")
                for fixed_frame, progressive_frame in zip(fixed, progressive):
                    writer.writerow({
                        "digit": digit,
                        "threshold": fixed_frame.threshold,
                        "fixed_iou": f"{fixed_frame.iou:.8f}",
                        "progressive_iou": f"{progressive_frame.iou:.8f}",
                        "fixed_matched_count": len(fixed_frame.matched_pixels),
                        "progressive_matched_count": len(progressive_frame.matched_pixels),
                        "fixed_retained_count": len(fixed_frame.retained_attachments),
                        "progressive_retained_count": len(progressive_frame.retained_attachments),
                        "progressive_offset_x": progressive_frame.offset_x,
                        "progressive_offset_y": progressive_frame.offset_y,
                        "progressive_rotation_degrees": progressive_frame.rotation_degrees,
                        "progressive_width_scale": progressive_frame.width_scale,
                        "progressive_height_scale": progressive_frame.height_scale,
                    })

        settings_path = run_directory / "experiment_settings.txt"
        settings_path.write_text(_settings_text(result.config), encoding="utf-8")
        zip_path = run_directory / "results.zip"
        files = sorted(path for path in run_directory.rglob("*") if path.is_file() and path != zip_path)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, path in enumerate(files):
                if cancel_event.is_set():
                    raise RecordingCancelled("Stage 1G packaging was canceled.")
                archive.write(path, path.relative_to(run_directory))
                if progress_callback is not None:
                    progress_callback(index + 1, len(files), f"Packaged {path.name}")
        return ProgressiveAnchoredRecordingResult(result, run_directory, len(result.frames))
    except Exception:
        shutil.rmtree(run_directory, ignore_errors=False)
        raise


def record_progressive_anchored_fit(
        config: ProgressiveAnchoredFitConfig,
        rgb_image: Image.Image,
        grayscale_image: Image.Image,
        output_root: Path | None = None,
        *,
        cancel_event: Event | None = None,
        progress_callback: ProgressCallback | None = None,
) -> ProgressiveAnchoredRecordingResult:
    if not config.auto_record:
        raise ValueError("Stage 1G recording requires auto_record=true.")
    cancellation = cancel_event or Event()
    analysis_progress = None
    if progress_callback is not None:
        callback = progress_callback

        def analysis_progress(done: int, total: int, message: str) -> None:
            callback(done, total, message)
    result = run_progressive_anchored_fit(
        config,
        rgb_image,
        grayscale_image,
        cancel_event=cancellation,
        progress_callback=analysis_progress,
    )
    root = output_root or (pictures_directory() / "PixelMathExperiments")
    return _write_recording(result, root, cancellation, progress_callback)
