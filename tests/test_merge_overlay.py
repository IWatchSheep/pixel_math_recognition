import unittest

from PIL import Image

from connected_sections import (
    MERGE_BOTH_COLOR,
    MERGE_PATH_ONLY_COLOR,
    MERGE_THRESHOLD_ONLY_COLOR,
    analyze_component_merge,
    create_merge_overlay,
)
from main import PixelMathRecognitionLab


class MergeOverlayTests(unittest.TestCase):
    def test_merge_views_highlight_path_without_modifying_original(self):
        grayscale = Image.new("L", (5, 2), 255)
        grayscale.putdata([0, 80, 100, 0, 0, 0, 0, 0, 0, 0])
        rgb = Image.new("RGB", (5, 2), (12, 34, 56))
        original_pixels = rgb.tobytes()

        result, overlay, merge_mask, categories = PixelMathRecognitionLab._merge_results(
            rgb,
            grayscale,
            50,
            100,
            ((0, 0), (3, 0)),
            (0, 0, 4, 1),
        )

        self.assertEqual(result.merge_threshold, 100)
        self.assertEqual(rgb.tobytes(), original_pixels)
        self.assertEqual(overlay.getpixel((1, 0)), (255, 255, 0))
        self.assertEqual(overlay.getpixel((2, 0)), MERGE_BOTH_COLOR)
        self.assertEqual(overlay.getpixel((4, 0)), (12, 34, 56))
        self.assertEqual(merge_mask.getpixel((2, 0)), 0)
        self.assertEqual(merge_mask.getpixel((4, 0)), 255)
        self.assertEqual(merge_mask.getpixel((0, 1)), 255)
        self.assertEqual(categories.new_at_threshold_only, ())
        self.assertEqual(categories.new_on_path_only, ((1, 0),))
        self.assertEqual(categories.both, ((2, 0),))

    def test_equal_brightness_alternate_bridges_have_deterministic_categories_and_colors(self):
        grayscale = Image.new("L", (5, 3), 255)
        for point, brightness in (
            ((0, 1), 0),
            ((4, 1), 0),
            ((1, 0), 80),
            ((2, 0), 100),
            ((3, 0), 80),
            ((1, 2), 100),
            ((2, 2), 100),
            ((3, 2), 100),
        ):
            grayscale.putpixel(point, brightness)
        rgb = Image.new("RGB", grayscale.size, (12, 34, 56))

        first_result = analyze_component_merge(
            grayscale, 10, (0, 1), (4, 1), max_threshold=150
        )
        first_overlay, first_categories = create_merge_overlay(rgb, first_result)
        second_result = analyze_component_merge(
            grayscale, 10, (0, 1), (4, 1), max_threshold=150
        )
        second_overlay, second_categories = create_merge_overlay(rgb, second_result)

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.merge_threshold, 100)
        self.assertEqual(
            first_result.newly_selected_pixels,
            ((2, 0), (1, 2), (2, 2), (3, 2)),
        )
        self.assertEqual(
            first_result.connecting_path,
            ((0, 1), (1, 0), (2, 0), (3, 0), (4, 1)),
        )
        self.assertEqual(
            first_result.bridge_pixels, ((1, 0), (2, 0), (3, 0))
        )
        self.assertEqual(first_categories.new_at_threshold_only, ((1, 2), (2, 2), (3, 2)))
        self.assertEqual(first_categories.new_on_path_only, ((1, 0), (3, 0)))
        self.assertEqual(first_categories.both, ((2, 0),))
        self.assertEqual(first_categories.newly_selected_total, 4)
        self.assertEqual(first_categories.path_pixels_total, 3)

        self.assertEqual(first_overlay.getpixel((1, 2)), MERGE_THRESHOLD_ONLY_COLOR)
        self.assertEqual(first_overlay.getpixel((1, 0)), MERGE_PATH_ONLY_COLOR)
        self.assertEqual(first_overlay.getpixel((2, 0)), MERGE_BOTH_COLOR)
        self.assertEqual(first_overlay.tobytes(), second_overlay.tobytes())
        self.assertEqual(first_categories, second_categories)

    def test_threshold_pixel_outside_representative_path_is_threshold_only(self):
        grayscale = Image.new("L", (5, 3), 255)
        for point, brightness in (
            ((0, 1), 0),
            ((4, 1), 0),
            ((1, 0), 80),
            ((2, 0), 100),
            ((3, 0), 80),
            ((1, 2), 100),
            ((2, 2), 100),
            ((3, 2), 100),
        ):
            grayscale.putpixel(point, brightness)
        rgb = Image.new("RGB", grayscale.size, (1, 2, 3))

        result = analyze_component_merge(grayscale, 10, (0, 1), (4, 1))
        overlay, categories = create_merge_overlay(rgb, result)

        self.assertIn((1, 2), result.newly_selected_pixels)
        self.assertNotIn((1, 2), result.connecting_path)
        self.assertIn((1, 2), categories.new_at_threshold_only)
        self.assertEqual(overlay.getpixel((1, 2)), MERGE_THRESHOLD_ONLY_COLOR)


if __name__ == "__main__":
    unittest.main()
