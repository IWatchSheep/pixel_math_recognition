import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pixel_processing import (
    create_binary_mask,
    get_pixel_info,
    grayscale_brightness,
    load_image,
    rgb_to_hex,
)


class PixelProcessingTests(unittest.TestCase):
    def test_load_image_returns_rgb_and_grayscale_with_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (3, 2), (25, 50, 75)).save(path)

            rgb, grayscale = load_image(path)

            self.assertEqual(rgb.mode, "RGB")
            self.assertEqual(grayscale.mode, "L")
            self.assertEqual(rgb.size, (3, 2))
            self.assertEqual(grayscale.size, (3, 2))

    def test_grayscale_brightness_for_primary_colors(self):
        self.assertEqual(grayscale_brightness(0, 0, 0), 0)
        self.assertEqual(grayscale_brightness(255, 255, 255), 255)
        self.assertEqual(grayscale_brightness(255, 0, 0), 76)
        self.assertEqual(grayscale_brightness(0, 255, 0), 150)
        self.assertEqual(grayscale_brightness(0, 0, 255), 29)

    def test_rgb_to_hex_formats_uppercase_two_digit_channels(self):
        self.assertEqual(rgb_to_hex(45, 48, 52), "#2D3034")

    def test_threshold_selects_values_at_or_below_threshold_as_black(self):
        grayscale = Image.new("L", (4, 1))
        grayscale.putdata([0, 80, 81, 255])

        result = create_binary_mask(grayscale, 80)

        self.assertEqual(list(result.tobytes()), [0, 0, 255, 255])

    def test_invalid_pixel_coordinates_are_rejected(self):
        rgb = Image.new("RGB", (2, 2), (1, 2, 3))
        gray = rgb.convert("L")
        for coordinates in ((-1, 0), (0, -1), (2, 0), (0, 2)):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    get_pixel_info(rgb, gray, *coordinates)

    def test_original_image_pixels_are_preserved_by_analysis(self):
        original_pixels = [(12, 34, 56), (200, 180, 160)]
        rgb = Image.new("RGB", (2, 1))
        rgb.putdata(original_pixels)
        before = rgb.tobytes()

        gray = rgb.convert("L")
        create_binary_mask(gray, 128)

        self.assertEqual(rgb.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
