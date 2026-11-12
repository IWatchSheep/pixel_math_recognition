"""Strict, data-only parser and runner for paste-and-run experiment settings."""

from dataclasses import dataclass
import math
import re

from PIL import Image

from pixel_processing import create_binary_mask
from template_matching import (
    AUTO_ALIGN_MAX_RADIUS,
    INDEPENDENT_SCALE_MAX,
    INDEPENDENT_SCALE_MIN,
    TEMPLATE_VARIANT_DIGIT,
    TEMPLATE_VARIANT_THIN_ZERO,
    TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
    AutoAlignResult,
    TemplateComparison,
    auto_align_offset,
    compare_template,
    create_template_overlay,
    load_template_variant,
)


TEMPLATE_NAMES = {
    "digit_5x7": TEMPLATE_VARIANT_DIGIT,
    "thin_zero": TEMPLATE_VARIANT_THIN_ZERO,
    "thin_zero_independent_xy": TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
}

REQUIRED_KEYS = frozenset(
    {
        "experiment",
        "roi_left",
        "roi_top",
        "roi_right",
        "roi_bottom",
        "brightness_threshold",
        "digit",
        "template",
        "uniform_scale",
        "width_scale",
        "height_scale",
        "offset_x",
        "offset_y",
        "auto_align_radius",
    }
)
INTEGER_KEYS = (
    "roi_left",
    "roi_top",
    "roi_right",
    "roi_bottom",
    "brightness_threshold",
    "digit",
    "offset_x",
    "offset_y",
    "auto_align_radius",
)
FLOAT_KEYS = ("uniform_scale", "width_scale", "height_scale")
INTEGER_PATTERN = re.compile(r"[+-]?\d+\Z")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    roi: tuple[int, int, int, int]
    brightness_threshold: int
    digit: str
    template_variant: str
    uniform_scale: float
    width_scale: float
    height_scale: float
    offset_x: int
    offset_y: int
    auto_align_radius: int


@dataclass(frozen=True)
class TemplateAutoAlignExperimentResult:
    config: ExperimentConfig
    image_size: tuple[int, int]
    initial_comparison: TemplateComparison
    alignment: AutoAlignResult
    overlay: Image.Image
    binary_mask: Image.Image


def parse_experiment_config(
    text: str, image_size: tuple[int, int]
) -> ExperimentConfig:
    """Parse and validate the complete allowed key=value configuration."""
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
        if key not in REQUIRED_KEYS:
            raise ValueError(f"Line {line_number}: unknown key '{key}'.")
        if key in values:
            raise ValueError(f"Line {line_number}: duplicate key '{key}'.")
        values[key] = value

    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}.")

    integers: dict[str, int] = {}
    for key in INTEGER_KEYS:
        value = values[key]
        if not INTEGER_PATTERN.fullmatch(value):
            raise ValueError(f"'{key}' must be an integer.")
        integers[key] = int(value)
    floats: dict[str, float] = {}
    for key in FLOAT_KEYS:
        try:
            value = float(values[key])
        except ValueError as error:
            raise ValueError(f"'{key}' must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"'{key}' must be a finite number.")
        floats[key] = value

    if values["experiment"] != "template_auto_align":
        raise ValueError("'experiment' must be 'template_auto_align'.")
    if not values["digit"].isdigit() or not 0 <= integers["digit"] <= 9:
        raise ValueError("'digit' must be one digit from 0 to 9.")
    template_name = values["template"]
    if template_name not in TEMPLATE_NAMES:
        raise ValueError(
            "'template' must be one of: " + ", ".join(sorted(TEMPLATE_NAMES)) + "."
        )
    template_variant = TEMPLATE_NAMES[template_name]
    if template_variant != TEMPLATE_VARIANT_DIGIT and integers["digit"] != 0:
        raise ValueError("The thin zero template variants require digit=0.")
    if not 0 <= integers["brightness_threshold"] <= 255:
        raise ValueError("'brightness_threshold' must be from 0 to 255.")
    if not 0.25 <= floats["uniform_scale"] <= 2.5:
        raise ValueError("'uniform_scale' must be between 0.25 and 2.5.")
    for key in ("width_scale", "height_scale"):
        if not INDEPENDENT_SCALE_MIN <= floats[key] <= INDEPENDENT_SCALE_MAX:
            raise ValueError(
                f"'{key}' must be between {INDEPENDENT_SCALE_MIN} and "
                f"{INDEPENDENT_SCALE_MAX}."
            )
    if not 0 <= integers["auto_align_radius"] <= AUTO_ALIGN_MAX_RADIUS:
        raise ValueError(
            f"'auto_align_radius' must be from 0 to {AUTO_ALIGN_MAX_RADIUS}."
        )

    left, top, right, bottom = (
        integers["roi_left"],
        integers["roi_top"],
        integers["roi_right"],
        integers["roi_bottom"],
    )
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"ROI must be inside the loaded {width}×{height} image."
        )
    if right <= left or bottom <= top:
        raise ValueError("ROI must have positive width and height.")

    return ExperimentConfig(
        values["experiment"],
        (left, top, right, bottom),
        integers["brightness_threshold"],
        str(integers["digit"]),
        template_variant,
        floats["uniform_scale"],
        floats["width_scale"],
        floats["height_scale"],
        integers["offset_x"],
        integers["offset_y"],
        integers["auto_align_radius"],
    )


def run_template_auto_align_experiment(
    config: ExperimentConfig,
    rgb_image: Image.Image,
    grayscale_image: Image.Image,
) -> TemplateAutoAlignExperimentResult:
    """Run a validated experiment on copies/derived masks, leaving inputs intact."""
    if rgb_image.mode != "RGB" or grayscale_image.mode != "L":
        raise ValueError("Experiment requires an RGB image and an L-mode grayscale image.")
    if rgb_image.size != grayscale_image.size:
        raise ValueError("RGB and grayscale image dimensions must match.")
    mask = create_binary_mask(grayscale_image, config.brightness_threshold)
    roi_mask = mask.crop(config.roi)
    template = load_template_variant(config.template_variant, config.digit)
    baseline_scale = min(
        roi_mask.width / template.width, roi_mask.height / template.height
    )
    width_scale = baseline_scale * config.uniform_scale * config.width_scale
    height_scale = baseline_scale * config.uniform_scale * config.height_scale
    initial = compare_template(
        roi_mask,
        template,
        config.offset_x,
        config.offset_y,
        width_scale,
        height_scale,
    )
    alignment = auto_align_offset(
        roi_mask,
        template,
        (config.offset_x, config.offset_y),
        config.auto_align_radius,
        width_scale,
        height_scale,
    )
    overlay = create_template_overlay(rgb_image, config.roi, alignment.comparison)
    return TemplateAutoAlignExperimentResult(
        config,
        rgb_image.size,
        initial,
        alignment,
        overlay,
        mask,
    )
