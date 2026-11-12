"""Deterministic history of an ROI component as brightness thresholds increase."""

from dataclasses import dataclass
from threading import Event
from typing import Callable

from PIL import Image

from connected_sections import Point, Roi, find_connected_sections

GROWTH_COMPONENT_COLOR = (0, 180, 0)
GROWTH_NEW_PIXELS_COLOR = (255, 128, 0)
GROWTH_BOUNDARY_COLOR = (0, 220, 255)
GROWTH_NEW_BOUNDARY_COLOR = (255, 0, 255)


class GrowthAnalysisCancelled(Exception):
    """Raised when a caller cancels an in-progress threshold analysis."""


@dataclass(frozen=True)
class ThresholdGrowthFrame:
    threshold: int
    seed: Point
    component_pixels: tuple[Point, ...]
    newly_admitted_pixels: tuple[Point, ...]
    pixel_count: int
    newly_admitted_count: int
    absorbed_start_sections: int
    bounding_rectangle: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    selected_pixel_count: int
    roi_pixel_count: int
    occupancy_percentage: float
    background_dominated: bool
    events: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdGrowthHistory:
    roi: Roi
    start_threshold: int
    max_threshold: int
    threshold_increment: int
    occupancy_limit_percentage: float
    start_section_count: int
    frames: tuple[ThresholdGrowthFrame, ...]


def _threshold_values(start: int, maximum: int, increment: int) -> tuple[int, ...]:
    values = list(range(start, maximum + 1, increment))
    if not values or values[-1] != maximum:
        values.append(maximum)
    return tuple(values)


def analyze_threshold_growth(
        grayscale_image: Image.Image,
        roi: Roi,
        seed: Point,
        start_threshold: int,
        max_threshold: int,
        threshold_increment: int = 1,
        occupancy_limit_percentage: float = 70.0,
        *,
        cancel_event: Event | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
) -> ThresholdGrowthHistory:
    """Track the seed's 8-connected component using original grayscale values.

    Threshold samples use the requested increment and always include the exact
    maximum threshold. The seed identifies one starting component; components
    absorbed later are counted by their starting labels, not renumbered identities.
    """
    if grayscale_image.mode != "L":
        raise ValueError("The image must be an 8-bit grayscale image (mode 'L').")
    if not isinstance(start_threshold, int) or not 0 <= start_threshold <= 255:
        raise ValueError("Starting threshold must be an integer from 0 to 255.")
    if not isinstance(max_threshold, int) or not 0 <= max_threshold <= 255:
        raise ValueError("Maximum threshold must be an integer from 0 to 255.")
    if max_threshold < start_threshold:
        raise ValueError("Maximum threshold must be at least the starting threshold.")
    if not isinstance(threshold_increment, int) or threshold_increment < 1:
        raise ValueError("Threshold increment must be a positive integer.")
    if not 0 < occupancy_limit_percentage <= 100:
        raise ValueError("Occupancy limit must be greater than 0 and at most 100.")

    image_width, image_height = grayscale_image.size
    left, top, right, bottom = roi
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError("ROI must be a non-empty rectangle within the image.")
    seed_x, seed_y = seed
    if not (left <= seed_x < right and top <= seed_y < bottom):
        raise ValueError("The selected seed must be inside the ROI.")

    crop = grayscale_image.crop(roi)
    width, height = crop.size
    brightness = crop.tobytes()
    seed_index = (seed_y - top) * width + (seed_x - left)
    if brightness[seed_index] > start_threshold:
        raise ValueError("The selected seed is not selected at the starting threshold.")

    start_mask = Image.frombytes(
        "L", crop.size,
        bytes(0 if value <= start_threshold else 255 for value in brightness),
    )
    start_sections, start_labels = find_connected_sections(start_mask)
    seed_label = start_labels[seed_index]
    if seed_label == 0:
        raise ValueError("The selected seed does not belong to a starting section.")

    frames: list[ThresholdGrowthFrame] = []
    previous_absorbed = 0
    previous_threshold = start_threshold
    roi_pixel_count = width * height

    thresholds = _threshold_values(
        start_threshold, max_threshold, threshold_increment
    )
    for frame_index, threshold in enumerate(thresholds, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise GrowthAnalysisCancelled("Threshold growth recording was canceled.")
        selected_mask = Image.frombytes(
            "L", crop.size,
            bytes(0 if value <= threshold else 255 for value in brightness),
        )
        _sections, labels = find_connected_sections(selected_mask)
        tracked_label = labels[seed_index]
        component_indices = {
            index for index, label in enumerate(labels) if label == tracked_label
        } if tracked_label else set()
        component_pixels = tuple(
            (index % width + left, index // width + top)
            for index in sorted(component_indices)
        )
        newly_admitted_indices = {
            index for index in component_indices
            if previous_threshold < brightness[index] <= threshold
        }
        newly_admitted_pixels = tuple(
            (index % width + left, index // width + top)
            for index in sorted(newly_admitted_indices)
        )
        absorbed_labels = {
            start_labels[index]
            for index in component_indices
            if start_labels[index] and start_labels[index] != seed_label
        }
        absorbed = len(absorbed_labels)
        selected_count = sum(value <= threshold for value in brightness)
        occupancy = 100.0 * selected_count / roi_pixel_count

        if component_pixels:
            xs = [point[0] for point in component_pixels]
            ys = [point[1] for point in component_pixels]
            bounds = (
                min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
            )
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            bounds = None
            centroid = None

        events = []
        if newly_admitted_pixels:
            events.append("New pixels extend the tracked section.")
        if absorbed > previous_absorbed:
            events.append(
                f"Starting sections became connected at threshold {threshold}."
            )
            events.append(
                f"Tracked section absorbed {absorbed - previous_absorbed} starting section(s)."
            )
        background_dominated = occupancy >= occupancy_limit_percentage
        if background_dominated and (
                not frames or not frames[-1].background_dominated
        ):
            events.append("Selected ROI occupancy reached the background-dominated limit.")

        frames.append(
            ThresholdGrowthFrame(
                threshold=threshold,
                seed=seed,
                component_pixels=component_pixels,
                newly_admitted_pixels=newly_admitted_pixels,
                pixel_count=len(component_indices),
                newly_admitted_count=len(newly_admitted_pixels),
                absorbed_start_sections=absorbed,
                bounding_rectangle=bounds,
                centroid=centroid,
                selected_pixel_count=selected_count,
                roi_pixel_count=roi_pixel_count,
                occupancy_percentage=occupancy,
                background_dominated=background_dominated,
                events=tuple(events),
            )
        )
        previous_absorbed = absorbed
        previous_threshold = threshold
        if progress_callback is not None:
            progress_callback(frame_index, len(thresholds), threshold)

    return ThresholdGrowthHistory(
        roi=roi,
        start_threshold=start_threshold,
        max_threshold=max_threshold,
        threshold_increment=threshold_increment,
        occupancy_limit_percentage=occupancy_limit_percentage,
        start_section_count=len(start_sections),
        frames=tuple(frames),
    )


def create_growth_overlay(
        rgb_image: Image.Image,
        frame: ThresholdGrowthFrame,
) -> Image.Image:
    """Show tracked pixels, new additions, and outline on an RGB copy."""
    if rgb_image.mode != "RGB":
        raise ValueError("The overlay source image must use RGB mode.")
    overlay = rgb_image.copy()
    component = set(frame.component_pixels)
    newly_admitted = set(frame.newly_admitted_pixels)
    width, height = rgb_image.size
    boundary = set()
    for x, y in component:
        if any(
                not (0 <= neighbor_x < width and 0 <= neighbor_y < height)
                or (neighbor_x, neighbor_y) not in component
                for neighbor_y in range(y - 1, y + 2)
                for neighbor_x in range(x - 1, x + 2)
        ):
            boundary.add((x, y))

    for point in component - boundary - newly_admitted:
        overlay.putpixel(point, GROWTH_COMPONENT_COLOR)
    for point in newly_admitted - boundary:
        overlay.putpixel(point, GROWTH_NEW_PIXELS_COLOR)
    for point in boundary - newly_admitted:
        overlay.putpixel(point, GROWTH_BOUNDARY_COLOR)
    for point in boundary & newly_admitted:
        overlay.putpixel(point, GROWTH_NEW_BOUNDARY_COLOR)
    return overlay
