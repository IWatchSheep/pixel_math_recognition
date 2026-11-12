"""Strict, data-only paste configuration for the Logical Grouping Lab."""

from dataclasses import dataclass
from array import array
import re

from PIL import Image

from connected_sections import ConnectedSection, Point, Roi, find_connected_sections
from logical_grouping import (
    GroupingAnalysis,
    LogicalPiece,
    analyze_grouping,
    combine_sections,
)
from pixel_processing import create_binary_mask
from connected_sections import colorize_sections

_INTEGER = re.compile(r"\d+\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
GROUPING_KEYS = frozenset({
    "experiment",
    "roi_left",
    "roi_top",
    "roi_right",
    "roi_bottom",
    "brightness_threshold",
    "pieces",
    "intended_groups",
})


@dataclass(frozen=True)
class LogicalGroupingConfig:
    experiment: str
    roi: Roi
    brightness_threshold: int
    pieces: tuple[tuple[str, tuple[Point, ...]], ...]
    intended_groups: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class LogicalGroupingExperimentResult:
    config: LogicalGroupingConfig
    analysis: GroupingAnalysis
    image_size: tuple[int, int]
    sections: tuple[ConnectedSection, ...]
    section_labels: array
    section_image: Image.Image


def _parse_pairs(
    value: str, kind: str, member_separator: str = ","
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    result = []
    seen = set()
    for entry in value.split(";"):
        if not entry or "=" not in entry:
            raise ValueError(f"Each {kind} entry must use name=members syntax.")
        name, raw_members = entry.split("=", 1)
        name = name.strip()
        members = tuple(member.strip() for member in raw_members.split(member_separator))
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"Invalid {kind} name '{name}'.")
        if name in seen:
            raise ValueError(f"Duplicate {kind} name '{name}'.")
        if not members or any(not member for member in members):
            raise ValueError(f"{kind.capitalize()} '{name}' must have members.")
        seen.add(name)
        result.append((name, members))
    return tuple(result)


def parse_logical_grouping_config(
    text: str,
    image_size: tuple[int, int],
) -> LogicalGroupingConfig:
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
        if key not in GROUPING_KEYS:
            raise ValueError(f"Line {line_number}: unknown key '{key}'.")
        if key in values:
            raise ValueError(f"Line {line_number}: duplicate key '{key}'.")
        values[key] = value
    missing = sorted(GROUPING_KEYS - values.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}.")
    if values["experiment"] != "logical_grouping":
        raise ValueError("'experiment' must be 'logical_grouping'.")

    integer_keys = (
        "roi_left", "roi_top", "roi_right", "roi_bottom", "brightness_threshold"
    )
    integers = {}
    for key in integer_keys:
        if not _INTEGER.fullmatch(values[key]):
            raise ValueError(f"'{key}' must be a non-negative integer.")
        integers[key] = int(values[key])
    left, top, right, bottom = (
        integers["roi_left"], integers["roi_top"],
        integers["roi_right"], integers["roi_bottom"],
    )
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"ROI must be inside the loaded {width}×{height} image.")
    if right <= left or bottom <= top:
        raise ValueError("ROI must have positive width and height.")
    threshold = integers["brightness_threshold"]
    if not 0 <= threshold <= 255:
        raise ValueError("'brightness_threshold' must be from 0 to 255.")

    piece_specs = _parse_pairs(values["pieces"], "piece", "|")
    pieces = []
    used_seeds = set()
    for identifier, members in piece_specs:
        seeds = []
        for member in members:
            coordinates = member.split(",")
            if len(coordinates) != 2 or any(
                not _INTEGER.fullmatch(coordinate.strip()) for coordinate in coordinates
            ):
                raise ValueError(
                    f"Piece '{identifier}' section references must be x,y seed coordinates."
                )
            point = (int(coordinates[0]), int(coordinates[1]))
            if point in used_seeds:
                raise ValueError(f"Section seed {point} is referenced more than once.")
            if not (left <= point[0] < right and top <= point[1] < bottom):
                raise ValueError(f"Section seed {point} must be inside the ROI.")
            used_seeds.add(point)
            seeds.append(point)
        pieces.append((identifier, tuple(seeds)))

    intended_groups = _parse_pairs(values["intended_groups"], "intended group")
    piece_names = {identifier for identifier, _seeds in pieces}
    assigned = []
    for group_name, members in intended_groups:
        unknown = set(members) - piece_names
        if unknown:
            raise ValueError(
                f"Intended group '{group_name}' references unknown piece(s): "
                + ", ".join(sorted(unknown)) + "."
            )
        assigned.extend(members)
    if set(assigned) != piece_names or len(assigned) != len(piece_names):
        raise ValueError("Intended groups must assign each configured piece exactly once.")

    return LogicalGroupingConfig(
        values["experiment"],
        (left, top, right, bottom),
        threshold,
        tuple(pieces),
        intended_groups,
    )


def run_logical_grouping_experiment(
    config: LogicalGroupingConfig,
    rgb_image: Image.Image,
    grayscale_image: Image.Image,
) -> LogicalGroupingExperimentResult:
    """Resolve coordinate-seeded sections and calculate geometry without mutation."""
    if rgb_image.mode != "RGB" or grayscale_image.mode != "L":
        raise ValueError("Grouping experiment requires RGB and L-mode images.")
    if rgb_image.size != grayscale_image.size:
        raise ValueError("RGB and grayscale image dimensions must match.")
    mask = create_binary_mask(grayscale_image, config.brightness_threshold).crop(config.roi)
    local_sections, labels = find_connected_sections(mask)
    left, top, _right, _bottom = config.roi
    full_sections = tuple(
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
    width = mask.width
    full_labels = array("i", [0]) * (rgb_image.width * rgb_image.height)
    for local_y in range(mask.height):
        source_start = local_y * width
        target_start = (local_y + top) * rgb_image.width + left
        full_labels[target_start:target_start + width] = labels[source_start:source_start + width]
    full_section_image = colorize_sections(full_labels, rgb_image.size)
    section_by_seed = {}
    for identifier, seeds in config.pieces:
        for seed in seeds:
            local_x, local_y = seed[0] - left, seed[1] - top
            label = labels[local_y * width + local_x]
            if not label:
                raise ValueError(
                    f"Seed {seed} is not selected at threshold {config.brightness_threshold}."
                )
            if label in section_by_seed:
                raise ValueError(
                    f"Seeds identify the same connected section at this threshold: {seed}."
                )
            section_by_seed[label] = seed

    pieces = []
    for identifier, seeds in config.pieces:
        resolved = []
        for seed in seeds:
            label = labels[(seed[1] - top) * width + seed[0] - left]
            resolved.append(full_sections[label - 1])
        pieces.append(combine_sections(identifier, tuple(resolved)))
    analysis = analyze_grouping(tuple(pieces))
    return LogicalGroupingExperimentResult(
        config, analysis, rgb_image.size, full_sections, full_labels, full_section_image
    )
