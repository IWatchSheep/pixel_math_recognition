"""Binary digit templates and deterministic template-mask comparison utilities."""

from dataclasses import dataclass
import math

from PIL import Image


DIGIT_TEMPLATE_ROWS = {
    "0": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}

TEMPLATE_VARIANT_DIGIT = "Digit template (selected digit)"
TEMPLATE_VARIANT_THIN_ZERO = "Thin zero outline (uniform)"
TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT = "Thin zero outline (independent X/Y)"
ZERO_THIN_TEMPLATE_ROWS = (
    "0011100",
    "0100010",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "1000001",
    "0100010",
    "0011100",
)
INDEPENDENT_SCALE_MIN = 0.8
INDEPENDENT_SCALE_MAX = 1.2
AUTO_ALIGN_MAX_RADIUS = 50

OVERLAP_COLOR = (0, 200, 0)
IMAGE_ONLY_COLOR = (255, 64, 64)
TEMPLATE_ONLY_COLOR = (64, 128, 255)


@dataclass(frozen=True)
class TemplateComparison:
    iou: float | None
    intersection_pixels: int
    image_only_pixels: int
    template_only_pixels: int
    union_pixels: int
    offset_x: int
    offset_y: int
    scale: float
    scale_y: float
    template_mask: Image.Image
    visualization: Image.Image


@dataclass
class TemplateTransform:
    """Uniform scale and translation relative to a centered ROI-fit baseline."""

    baseline_scale: float
    offset_x: int = 0
    offset_y: int = 0
    scale_multiplier: float = 1.0
    width_scale_multiplier: float = 1.0
    height_scale_multiplier: float = 1.0

    @property
    def scale(self) -> float:
        return (
            self.baseline_scale
            * self.scale_multiplier
            * self.width_scale_multiplier
        )

    @property
    def scale_y(self) -> float:
        return (
            self.baseline_scale
            * self.scale_multiplier
            * self.height_scale_multiplier
        )

    def set_independent_scales(self, width: float, height: float) -> None:
        for value in (width, height):
            if not math.isfinite(value) or not (
                INDEPENDENT_SCALE_MIN <= value <= INDEPENDENT_SCALE_MAX
            ):
                raise ValueError(
                    "Independent width and height scales must be between 0.8 and 1.2."
                )
        self.width_scale_multiplier = width
        self.height_scale_multiplier = height

    def reset(self) -> None:
        self.offset_x = 0
        self.offset_y = 0
        self.scale_multiplier = 1.0
        self.width_scale_multiplier = 1.0
        self.height_scale_multiplier = 1.0


@dataclass(frozen=True)
class TemplateFitSettings:
    variant: str
    digit: str
    threshold: int
    offset_x: int
    offset_y: int
    scale_multiplier: float
    width_scale_multiplier: float
    height_scale_multiplier: float


@dataclass(frozen=True)
class AutoAlignResult:
    starting_offset: tuple[int, int]
    starting_iou: float | None
    selected_offset: tuple[int, int]
    selected_iou: float | None
    improvement_percentage_points: float | None
    comparison: TemplateComparison
    search_radius: int


def transfer_template_fit(
    settings: TemplateFitSettings, new_baseline_scale: float
) -> TemplateTransform:
    """Recenter fit settings on a new ROI while preserving relative parameters."""
    if not math.isfinite(new_baseline_scale) or new_baseline_scale <= 0:
        raise ValueError("New ROI baseline scale must be finite and positive.")
    transform = TemplateTransform(
        baseline_scale=new_baseline_scale,
        offset_x=settings.offset_x,
        offset_y=settings.offset_y,
        scale_multiplier=settings.scale_multiplier,
    )
    transform.set_independent_scales(
        settings.width_scale_multiplier, settings.height_scale_multiplier
    )
    return transform


def format_roi_coordinates(roi: tuple[int, int, int, int]) -> str:
    """Format the application's half-open (left, top, right, bottom) ROI."""
    left, top, right, bottom = roi
    return (
        f"left={left}, top={top}, right={right}, bottom={bottom} "
        f"(right/bottom exclusive)"
    )


def load_digit_template(digit: str) -> Image.Image:
    """Load a 5x7 manually defined digit mask; black pixels are selected."""
    try:
        rows = DIGIT_TEMPLATE_ROWS[digit]
    except KeyError as error:
        raise ValueError("Digit template must be one of 0 through 9.") from error
    width = len(rows[0])
    if len(rows) != 7 or any(len(row) != width for row in rows):
        raise ValueError("Digit template data must be a rectangular 7-row bitmap.")
    if any(pixel not in "01" for row in rows for pixel in row):
        raise ValueError("Digit template data may contain only '0' and '1'.")
    return Image.frombytes(
        "L",
        (width, len(rows)),
        bytes(0 if pixel == "1" else 255 for row in rows for pixel in row),
    )


def load_template_variant(variant: str, digit: str = "0") -> Image.Image:
    """Load a manually authored template variant without altering the digit bank."""
    if variant == TEMPLATE_VARIANT_DIGIT:
        return load_digit_template(digit)
    if variant in (
        TEMPLATE_VARIANT_THIN_ZERO,
        TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
    ):
        if digit != "0":
            raise ValueError("The thin zero template variants are only available for digit 0.")
        return Image.frombytes(
            "L",
            (len(ZERO_THIN_TEMPLATE_ROWS[0]), len(ZERO_THIN_TEMPLATE_ROWS)),
            bytes(
                0 if pixel == "1" else 255
                for row in ZERO_THIN_TEMPLATE_ROWS
                for pixel in row
            ),
        )
    raise ValueError(f"Unknown template variant: {variant}")


def scale_binary_template(
    template: Image.Image, scale: float, scale_y: float | None = None
) -> Image.Image:
    """Resize a binary template, uniformly by default, using nearest-neighbor sampling."""
    if template.mode != "L":
        raise ValueError("Template must be an 8-bit grayscale binary mask.")
    vertical_scale = scale if scale_y is None else scale_y
    if (
        not math.isfinite(scale)
        or scale <= 0
        or not math.isfinite(vertical_scale)
        or vertical_scale <= 0
    ):
        raise ValueError("Template scales must be finite positive numbers.")
    width, height = template.size
    scaled_size = (
        max(1, round(width * scale)),
        max(1, round(height * vertical_scale)),
    )
    resized = template.resize(scaled_size, Image.Resampling.NEAREST)
    return resized.point([0 if value < 128 else 255 for value in range(256)])


def compare_template(
    image_mask: Image.Image,
    template: Image.Image,
    offset_x: int = 0,
    offset_y: int = 0,
    scale: float = 1.0,
    scale_y: float | None = None,
) -> TemplateComparison:
    """Center a scaled template in the image mask and calculate its pixel IoU.

    ``scale_y`` defaults to ``scale`` for uniform scaling. Offsets translate the
    template from the centered baseline, in image pixels.
    A zero union produces ``iou=None`` (undefined); a one-sided empty mask scores 0.
    """
    if image_mask.mode != "L" or template.mode != "L":
        raise ValueError("Image and template masks must be 8-bit grayscale images.")
    if image_mask.width < 1 or image_mask.height < 1:
        raise ValueError("Image mask must have positive dimensions.")
    vertical_scale = scale if scale_y is None else scale_y
    scaled = scale_binary_template(template, scale, vertical_scale)
    origin_x = (image_mask.width - scaled.width) // 2 + offset_x
    origin_y = (image_mask.height - scaled.height) // 2 + offset_y

    placed = Image.new("L", image_mask.size, 255)
    left = max(0, origin_x)
    top = max(0, origin_y)
    right = min(image_mask.width, origin_x + scaled.width)
    bottom = min(image_mask.height, origin_y + scaled.height)
    if left < right and top < bottom:
        placed.paste(
            scaled.crop((left - origin_x, top - origin_y, right - origin_x, bottom - origin_y)),
            (left, top),
        )

    image_selected = image_mask.tobytes()
    template_selected = placed.tobytes()
    overlap = bytearray(len(image_selected) * 3)
    intersection = image_only = template_only = union = 0
    for index, (image_value, template_value) in enumerate(
        zip(image_selected, template_selected)
    ):
        is_image = image_value == 0
        is_template = template_value == 0
        offset = index * 3
        if is_image and is_template:
            color = OVERLAP_COLOR
            intersection += 1
            union += 1
        elif is_image:
            color = IMAGE_ONLY_COLOR
            image_only += 1
            union += 1
        elif is_template:
            color = TEMPLATE_ONLY_COLOR
            template_only += 1
            union += 1
        else:
            color = (0, 0, 0)
        overlap[offset : offset + 3] = bytes(color)

    visualization = Image.frombytes("RGB", image_mask.size, bytes(overlap))
    iou = intersection / union if union else None
    return TemplateComparison(
        iou,
        intersection,
        image_only,
        template_only,
        union,
        offset_x,
        offset_y,
        scale,
        vertical_scale,
        placed,
        visualization,
    )


def auto_align_offset(
    image_mask: Image.Image,
    template: Image.Image,
    starting_offset: tuple[int, int],
    search_radius: int,
    scale: float = 1.0,
    scale_y: float | None = None,
) -> AutoAlignResult:
    """Find the best integer translation in a bounded square neighborhood.

    Equal IoU scores prefer the least squared displacement from the starting
    offset, then the smaller Y offset, then the smaller X offset.
    """
    if not isinstance(search_radius, int) or not (
        0 <= search_radius <= AUTO_ALIGN_MAX_RADIUS
    ):
        raise ValueError(
            f"Search radius must be an integer from 0 to {AUTO_ALIGN_MAX_RADIUS}."
        )
    start_x, start_y = starting_offset
    starting = compare_template(
        image_mask, template, start_x, start_y, scale, scale_y
    )
    def candidate_rank(comparison: TemplateComparison) -> tuple[float, int, int, int]:
        score = comparison.iou if comparison.iou is not None else -1.0
        offset_x, offset_y = comparison.offset_x, comparison.offset_y
        displacement_squared = (
            (offset_x - start_x) ** 2 + (offset_y - start_y) ** 2
        )
        return -score, displacement_squared, offset_y, offset_x

    best = starting
    best_rank = candidate_rank(starting)
    for offset_y in range(start_y - search_radius, start_y + search_radius + 1):
        for offset_x in range(start_x - search_radius, start_x + search_radius + 1):
            comparison = compare_template(
                image_mask, template, offset_x, offset_y, scale, scale_y
            )
            rank = candidate_rank(comparison)
            if rank < best_rank:
                best, best_rank = comparison, rank
    selected = best
    improvement = (
        None
        if starting.iou is None or selected.iou is None
        else (selected.iou - starting.iou) * 100
    )
    return AutoAlignResult(
        starting_offset,
        starting.iou,
        (selected.offset_x, selected.offset_y),
        selected.iou,
        improvement,
        selected,
        search_radius,
    )


def create_template_overlay(
    rgb_image: Image.Image,
    roi: tuple[int, int, int, int],
    comparison: TemplateComparison,
) -> Image.Image:
    """Color selected-pixel agreement on an RGB copy without changing the source."""
    if rgb_image.mode != "RGB":
        raise ValueError("Source image must be RGB.")
    left, top, right, bottom = roi
    if (
        left < 0 or top < 0 or right > rgb_image.width or bottom > rgb_image.height
        or right <= left or bottom <= top
        or comparison.visualization.size != (right - left, bottom - top)
    ):
        raise ValueError("ROI must be a non-empty in-bounds match region.")
    region = rgb_image.crop(roi)
    pixels = bytearray(region.tobytes())
    colors = comparison.visualization.tobytes()
    for offset in range(0, len(colors), 3):
        if colors[offset : offset + 3] != b"\x00\x00\x00":
            pixels[offset : offset + 3] = colors[offset : offset + 3]
    region = Image.frombytes("RGB", region.size, bytes(pixels))
    result = rgb_image.copy()
    result.paste(region, (left, top))
    return result
