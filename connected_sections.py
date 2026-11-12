"""Deterministic 8-neighbor connected-component analysis."""

from array import array
from dataclasses import dataclass

from PIL import Image

Roi = tuple[int, int, int, int]
Point = tuple[int, int]
MERGE_THRESHOLD_ONLY_COLOR = (255, 0, 255)
MERGE_PATH_ONLY_COLOR = (255, 255, 0)
MERGE_BOTH_COLOR = (0, 255, 255)


@dataclass(frozen=True)
class ConnectedSection:
    number: int
    pixel_count: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def bounding_rectangle(self) -> tuple[int, int, int, int]:
        """Return (left, top, width, height); extrema are inclusive."""
        return (
            self.min_x,
            self.min_y,
            self.max_x - self.min_x + 1,
            self.max_y - self.min_y + 1,
        )


@dataclass(frozen=True)
class MergeAnalysis:
    start_threshold: int
    max_threshold: int
    seed_a: Point
    seed_b: Point
    seed_section_sizes: tuple[int, int]
    merge_threshold: int | None
    newly_selected_pixels: tuple[Point, ...]
    connecting_path: tuple[Point, ...]
    bridge_pixels: tuple[Point, ...]
    merged_component_size: int
    absorbed_start_sections: int
    selected_pixel_count: int
    roi_pixel_count: int
    roi: Roi


@dataclass(frozen=True)
class MergeOverlayCategories:
    new_at_threshold_only: tuple[Point, ...]
    new_on_path_only: tuple[Point, ...]
    both: tuple[Point, ...]

    @property
    def newly_selected_total(self) -> int:
        return len(self.new_at_threshold_only) + len(self.both)

    @property
    def path_pixels_total(self) -> int:
        return len(self.new_on_path_only) + len(self.both)


def find_connected_sections(mask: Image.Image) -> tuple[list[ConnectedSection], array]:
    """Find black-pixel components with 8-neighbor connectivity.

    The returned integer label array uses zero for background and section numbers
    starting at one. Sections are numbered by their first row-major pixel.
    """
    if mask.mode != "L":
        raise ValueError("The mask must be an 8-bit grayscale image (mode 'L').")

    width, height = mask.size
    pixel_count = width * height
    pixels = mask.tobytes()
    labels = array("i", [0]) * pixel_count
    sections: list[ConnectedSection] = []
    stack: list[int] = []
    section_number = 0

    for start in range(pixel_count):
        if pixels[start] != 0 or labels[start] != 0:
            continue

        section_number += 1
        labels[start] = section_number
        stack.append(start)
        count = 0
        min_x = max_x = start % width
        min_y = max_y = start // width

        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row_start = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row_start + neighbor_x
                    if (
                            pixels[neighbor] == 0
                            and labels[neighbor] == 0
                    ):
                        labels[neighbor] = section_number
                        stack.append(neighbor)

        sections.append(
            ConnectedSection(
                section_number, count, min_x, max_x, min_y, max_y
            )
        )

    return sections, labels


def analyze_component_merge(
        grayscale_image: Image.Image,
        start_threshold: int,
        seed_a: Point,
        seed_b: Point,
        max_threshold: int = 255,
        roi: Roi | None = None,
) -> MergeAnalysis:
    """Track two starting components and find their first 8-connected merge.

    ROI coordinates and returned points are in the full-image coordinate system.
    ROI bounds are half-open: (left, top, right, bottom). For equal-length paths,
    breadth-first search uses row-major neighbor order for deterministic output.
    """
    if grayscale_image.mode != "L":
        raise ValueError("The image must be an 8-bit grayscale image (mode 'L').")
    if not 0 <= start_threshold <= 255 or not 0 <= max_threshold <= 255:
        raise ValueError("Thresholds must be integers from 0 to 255.")
    if start_threshold > max_threshold:
        raise ValueError("Maximum threshold must be at least the starting threshold.")

    width, height = grayscale_image.size
    bounds = roi or (0, 0, width, height)
    left, top, right, bottom = bounds
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("ROI must be a non-empty rectangle within the image.")
    for seed in (seed_a, seed_b):
        x, y = seed
        if not (left <= x < right and top <= y < bottom):
            raise ValueError("Both seed pixels must be inside the ROI.")

    crop = grayscale_image.crop(bounds)
    crop_width, crop_height = crop.size
    brightness = crop.tobytes()
    start_mask = Image.frombytes(
        "L",
        crop.size,
        bytes(0 if value <= start_threshold else 255 for value in brightness),
    )
    sections, labels = find_connected_sections(start_mask)
    local_a = (seed_a[0] - left, seed_a[1] - top)
    local_b = (seed_b[0] - left, seed_b[1] - top)
    index_a = local_a[1] * crop_width + local_a[0]
    index_b = local_b[1] * crop_width + local_b[0]
    label_a, label_b = labels[index_a], labels[index_b]
    if not label_a or not label_b:
        raise ValueError("Both seed pixels must be selected at the starting threshold.")

    section_sizes = {section.number: section.pixel_count for section in sections}
    sizes = (section_sizes[label_a], section_sizes[label_b])

    def search(threshold: int) -> tuple[list[int], set[int]] | None:
        parents = {index_a: -1}
        pending = [index_a]
        reached = {index_a}
        cursor = 0
        target_found = index_a == index_b
        while cursor < len(pending):
            index = pending[cursor]
            cursor += 1
            x, y = index % crop_width, index // crop_width
            for neighbor_y in range(max(0, y - 1), min(crop_height, y + 2)):
                row_start = neighbor_y * crop_width
                for neighbor_x in range(max(0, x - 1), min(crop_width, x + 2)):
                    neighbor = row_start + neighbor_x
                    if neighbor in reached or brightness[neighbor] > threshold:
                        continue
                    reached.add(neighbor)
                    parents[neighbor] = index
                    pending.append(neighbor)
                    if neighbor == index_b:
                        target_found = True
        if not target_found:
            return None
        path = [index_b]
        while path[-1] != index_a:
            path.append(parents[path[-1]])
        path.reverse()
        return path, reached

    merge_threshold: int | None = None
    found = search(start_threshold)
    if found is not None:
        merge_threshold = start_threshold
    else:
        for threshold in sorted(
                {value for value in brightness if start_threshold < value <= max_threshold}
        ):
            found = search(threshold)
            if found is not None:
                merge_threshold = threshold
                break

    if found is None or merge_threshold is None:
        selected_at_limit = sum(value <= max_threshold for value in brightness)
        return MergeAnalysis(
            start_threshold, max_threshold, seed_a, seed_b, sizes, None,
            (), (), (), 0, 0, selected_at_limit, len(brightness), bounds,
        )

    path_indices, component_indices = found
    path = tuple(
        (index % crop_width + left, index // crop_width + top)
        for index in path_indices
    )
    bridge_pixels = tuple(
        point
        for point, index in zip(path, path_indices)
        if brightness[index] > start_threshold
    )
    newly_selected = tuple(
        (index % crop_width + left, index // crop_width + top)
        for index, value in enumerate(brightness)
        if value == merge_threshold
    ) if merge_threshold > start_threshold else ()
    start_labels = {labels[index] for index in component_indices if labels[index]}
    absorbed = len(start_labels - {label_a, label_b})

    return MergeAnalysis(
        start_threshold,
        max_threshold,
        seed_a,
        seed_b,
        sizes,
        merge_threshold,
        newly_selected,
        path,
        bridge_pixels,
        len(component_indices),
        absorbed,
        sum(value <= merge_threshold for value in brightness),
        len(brightness),
        bounds,
    )


def create_merge_overlay(
    rgb_image: Image.Image, result: MergeAnalysis
) -> tuple[Image.Image, MergeOverlayCategories]:
    """Color three disjoint merge-overlay pixel categories on an RGB copy."""
    if rgb_image.mode != "RGB":
        raise ValueError("The overlay source image must use RGB mode.")

    newly_selected = set(result.newly_selected_pixels)
    path_pixels = set(result.bridge_pixels)
    categories = MergeOverlayCategories(
        tuple(sorted(newly_selected - path_pixels, key=lambda point: (point[1], point[0]))),
        tuple(sorted(path_pixels - newly_selected, key=lambda point: (point[1], point[0]))),
        tuple(sorted(newly_selected & path_pixels, key=lambda point: (point[1], point[0]))),
    )
    overlay = rgb_image.copy()
    for point in categories.new_at_threshold_only:
        overlay.putpixel(point, MERGE_THRESHOLD_ONLY_COLOR)
    for point in categories.new_on_path_only:
        overlay.putpixel(point, MERGE_PATH_ONLY_COLOR)
    for point in categories.both:
        overlay.putpixel(point, MERGE_BOTH_COLOR)
    return overlay, categories


def colorize_sections(labels: array, size: tuple[int, int]) -> Image.Image:
    """Create a deterministic RGB image with each section in a distinct color."""
    width, height = size
    if len(labels) != width * height:
        raise ValueError("Label count does not match image dimensions.")

    colors = [(255, 255, 255)]
    largest_label = max(labels, default=0)
    for number in range(1, largest_label + 1):
        hue = (number * 0.618033988749895) % 1.0
        sector = int(hue * 6)
        fraction = hue * 6 - sector
        value = 0.9
        low = 0.18
        rising = value * (1 - fraction)
        falling = low + (value - low) * fraction
        rgb = (
            (value, rising, low),
            (falling, value, low),
            (low, value, rising),
            (low, falling, value),
            (rising, low, value),
            (value, low, falling),
        )[sector % 6]
        colors.append(tuple(round(channel * 255) for channel in rgb))

    output = bytearray(width * height * 3)
    for index, label in enumerate(labels):
        red, green, blue = colors[label]
        offset = index * 3
        output[offset: offset + 3] = bytes((red, green, blue))
    return Image.frombytes("RGB", size, bytes(output))
