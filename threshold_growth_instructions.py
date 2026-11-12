"""Strict, data-only paste configuration for Threshold Growth History."""

from dataclasses import dataclass
import math
import re
from array import array
from threading import Event
from typing import Callable

from PIL import Image

from connected_sections import ConnectedSection, Point, Roi, colorize_sections, find_connected_sections
from pixel_processing import create_binary_mask
from threshold_growth import (
    ThresholdGrowthHistory,
    analyze_threshold_growth,
)

_INTEGER = re.compile(r"\d+\Z")
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)\Z")
REQUIRED_KEYS = frozenset({
    "experiment",
    "roi_left",
    "roi_top",
    "roi_right",
    "roi_bottom",
    "brightness_threshold",
    "maximum_threshold",
    "threshold_increment",
    "background_limit_percent",
})
OPTIONAL_KEYS = frozenset({
    "seed_x",
    "seed_y",
    "auto_record",
    "auto_zoom",
    "snapshot_mode",
    "experiment_name",
})
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


@dataclass(frozen=True)
class ThresholdGrowthConfig:
    experiment: str
    roi: Roi
    brightness_threshold: int
    maximum_threshold: int
    threshold_increment: int
    background_limit_percentage: float
    seed: Point | None
    auto_record: bool = False
    auto_zoom: bool = False
    snapshot_mode: str = "on_change"
    experiment_name: str = "threshold_growth"


@dataclass(frozen=True)
class ThresholdGrowthExperimentResult:
    config: ThresholdGrowthConfig
    image_size: tuple[int, int]
    sections: tuple[ConnectedSection, ...]
    section_labels: array
    section_image: Image.Image
    history: ThresholdGrowthHistory | None


def parse_threshold_growth_config(
    text: str,
    image_size: tuple[int, int],
) -> ThresholdGrowthConfig:
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"Line {line_number}: expected key=value.")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or not value:
            raise ValueError(f"Line {line_number}: key and value must be non-empty.")
        if key not in ALLOWED_KEYS:
            raise ValueError(f"Line {line_number}: unknown key '{key}'.")
        if key in values:
            raise ValueError(f"Line {line_number}: duplicate key '{key}'.")
        values[key] = value

    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}.")
    if values["experiment"] != "threshold_growth":
        raise ValueError("'experiment' must be 'threshold_growth'.")
    if ("seed_x" in values) != ("seed_y" in values):
        raise ValueError("'seed_x' and 'seed_y' must be supplied together.")
    auto_record = _parse_boolean(values.get("auto_record", "false"), "auto_record")
    auto_zoom = _parse_boolean(values.get("auto_zoom", "false"), "auto_zoom")
    snapshot_mode = values.get("snapshot_mode", "on_change")
    if snapshot_mode != "on_change":
        raise ValueError("'snapshot_mode' must be 'on_change'.")
    experiment_name = values.get("experiment_name", "threshold_growth")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", experiment_name):
        raise ValueError(
            "'experiment_name' must be 1–48 letters, digits, underscores, or "
            "hyphens, and start with a letter or digit."
        )
    if auto_record and "seed_x" not in values:
        raise ValueError("'auto_record=true' requires explicit seed_x and seed_y.")

    integer_keys = (
        "roi_left",
        "roi_top",
        "roi_right",
        "roi_bottom",
        "brightness_threshold",
        "maximum_threshold",
        "threshold_increment",
    )
    integers = {}
    for key in integer_keys:
        if not _INTEGER.fullmatch(values[key]):
            raise ValueError(f"'{key}' must be a non-negative integer.")
        integers[key] = int(values[key])

    if not _NUMBER.fullmatch(values["background_limit_percent"]):
        raise ValueError("'background_limit_percent' must be a finite number.")
    occupancy_limit = float(values["background_limit_percent"])
    if not math.isfinite(occupancy_limit) or not 0 < occupancy_limit <= 100:
        raise ValueError("'background_limit_percent' must be greater than 0 and at most 100.")
    start = integers["brightness_threshold"]
    maximum = integers["maximum_threshold"]
    increment = integers["threshold_increment"]
    if not 0 <= start <= 255:
        raise ValueError("'brightness_threshold' must be from 0 to 255.")
    if not start <= maximum <= 255:
        raise ValueError("'maximum_threshold' must be from the starting threshold to 255.")
    if not 1 <= increment <= 255:
        raise ValueError("'threshold_increment' must be from 1 to 255.")

    left, top, right, bottom = (
        integers["roi_left"],
        integers["roi_top"],
        integers["roi_right"],
        integers["roi_bottom"],
    )
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"ROI must be inside the loaded {width}×{height} image.")
    if right <= left or bottom <= top:
        raise ValueError("ROI must have positive width and height.")

    seed = None
    if "seed_x" in values:
        if not _INTEGER.fullmatch(values["seed_x"]) or not _INTEGER.fullmatch(values["seed_y"]):
            raise ValueError("'seed_x' and 'seed_y' must be non-negative integer coordinates.")
        seed = (int(values["seed_x"]), int(values["seed_y"]))
        if not (left <= seed[0] < right and top <= seed[1] < bottom):
            raise ValueError("Seed coordinates must be inside the ROI.")

    return ThresholdGrowthConfig(
        values["experiment"],
        (left, top, right, bottom),
        start,
        maximum,
        increment,
        occupancy_limit,
        seed,
        auto_record,
        auto_zoom,
        snapshot_mode,
        experiment_name,
    )


def _parse_boolean(value: str, name: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"'{name}' must be 'true' or 'false'.")
    return value == "true"


def run_threshold_growth_experiment(
    config: ThresholdGrowthConfig,
    rgb_image: Image.Image,
    grayscale_image: Image.Image,
    *,
    cancel_event: Event | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> ThresholdGrowthExperimentResult:
    """Find ROI components and optionally track the explicitly seeded component."""
    if rgb_image.mode != "RGB" or grayscale_image.mode != "L":
        raise ValueError("Threshold growth requires RGB and L-mode images.")
    if rgb_image.size != grayscale_image.size:
        raise ValueError("RGB and grayscale image dimensions must match.")

    mask = create_binary_mask(grayscale_image, config.brightness_threshold).crop(config.roi)
    local_sections, local_labels = find_connected_sections(mask)
    left, top, _right, _bottom = config.roi
    sections = tuple(
        ConnectedSection(
            section.number,
            section.pixel_count,
            section.min_x + left,
            section.max_x + left,
            section.min_y + top,
            section.max_y + top,
        )
        for section in local_sections
    )
    image_width, image_height = rgb_image.size
    labels = array("i", [0]) * (image_width * image_height)
    for local_y in range(mask.height):
        source_start = local_y * mask.width
        target_start = (local_y + top) * image_width + left
        labels[target_start:target_start + mask.width] = (
            local_labels[source_start:source_start + mask.width]
        )
    section_image = colorize_sections(labels, rgb_image.size)

    history = None
    if config.seed is not None:
        seed_x, seed_y = config.seed
        label = labels[seed_y * image_width + seed_x]
        if not label:
            raise ValueError(
                f"Seed {config.seed} is not selected at threshold "
                f"{config.brightness_threshold}."
            )
        history = analyze_threshold_growth(
            grayscale_image,
            config.roi,
            config.seed,
            config.brightness_threshold,
            config.maximum_threshold,
            config.threshold_increment,
            config.background_limit_percentage,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )

    return ThresholdGrowthExperimentResult(
        config,
        rgb_image.size,
        sections,
        labels,
        section_image,
        history,
    )
