"""Image loading and pixel-level brightness utilities."""

from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: str | Path) -> tuple[Image.Image, Image.Image]:
    """Load an image as separate RGB and grayscale images without editing the source."""
    with Image.open(path) as source:
        if source.format is None or f".{source.format.lower()}" not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported image format.")
        rgb_image = source.convert("RGB")
    return rgb_image, rgb_image.convert("L")


def create_preview(
    rgb_image: Image.Image, max_size: tuple[int, int]
) -> Image.Image:
    """Create a downscaled RGB preview without changing the source pixel image."""
    max_width, max_height = max_size
    scale = min(max_width / rgb_image.width, max_height / rgb_image.height, 1.0)
    size = (
        max(1, round(rgb_image.width * scale)),
        max(1, round(rgb_image.height * scale)),
    )
    return rgb_image.resize(size, Image.Resampling.LANCZOS)


def grayscale_brightness(red: int, green: int, blue: int) -> int:
    """Return Pillow-compatible 8-bit grayscale brightness for an RGB color."""
    for channel in (red, green, blue):
        if not isinstance(channel, int) or not 0 <= channel <= 255:
            raise ValueError("RGB channels must be integers from 0 to 255.")
    return (red * 19595 + green * 38470 + blue * 7471 + 32768) >> 16


def rgb_to_hex(red: int, green: int, blue: int) -> str:
    """Format three 8-bit RGB channel values as a hexadecimal color."""
    for channel in (red, green, blue):
        if not isinstance(channel, int) or not 0 <= channel <= 255:
            raise ValueError("RGB channels must be integers from 0 to 255.")
    return f"#{red:02X}{green:02X}{blue:02X}"


def get_pixel_info(
    rgb_image: Image.Image, grayscale_image: Image.Image, x: int, y: int
) -> dict[str, int | str]:
    """Return original RGB, hex, and grayscale values at one image coordinate."""
    width, height = rgb_image.size
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Pixel coordinates must be within 0..{width - 1}, 0..{height - 1}.")
    red, green, blue = rgb_image.getpixel((x, y))
    return {
        "x": x,
        "y": y,
        "red": red,
        "green": green,
        "blue": blue,
        "hex": rgb_to_hex(red, green, blue),
        "brightness": grayscale_image.getpixel((x, y)),
    }


def create_binary_mask(grayscale_image: Image.Image, threshold: int) -> Image.Image:
    """Return an L-mode mask: selected pixels are black, excluded pixels white."""
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError("Brightness threshold must be an integer from 0 to 255.")
    lookup = [0 if brightness <= threshold else 255 for brightness in range(256)]
    return grayscale_image.point(lookup)
