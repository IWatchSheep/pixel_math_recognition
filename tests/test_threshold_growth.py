import unittest

from PIL import Image

from threshold_growth import (
    GROWTH_BOUNDARY_COLOR,
    GROWTH_COMPONENT_COLOR,
    GROWTH_NEW_PIXELS_COLOR,
    analyze_threshold_growth,
    create_growth_overlay,
)


class ThresholdGrowthTests(unittest.TestCase):
    def test_gradual_growth_and_newly_admitted_pixels(self):
        image = Image.new("L", (7, 3), 255)
        image.putdata([
            255, 255, 255, 255, 255, 255, 255,
            255, 0, 20, 30, 30, 0, 255,
            255, 255, 255, 255, 255, 255, 255,
        ])

        result = analyze_threshold_growth(
            image, (0, 0, 7, 3), (1, 1), 0, 30, threshold_increment=20
        )

        self.assertEqual([frame.pixel_count for frame in result.frames], [1, 2, 5])
        self.assertEqual([frame.newly_admitted_count for frame in result.frames], [0, 1, 2])
        self.assertEqual(result.frames[1].newly_admitted_pixels, ((2, 1),))
        self.assertEqual(result.frames[2].absorbed_start_sections, 1)
        self.assertIn("Starting sections became connected at threshold 30.",
                      result.frames[2].events)
        self.assertIn("Tracked section absorbed 1 starting section(s).",
                      result.frames[2].events)

    def test_equal_brightness_pixels_are_admitted_together(self):
        image = Image.new("L", (5, 1))
        image.putdata([0, 80, 80, 80, 255])

        result = analyze_threshold_growth(image, (0, 0, 5, 1), (0, 0), 0, 90)

        self.assertEqual(len(result.frames), 91)
        self.assertEqual(result.frames[-1].newly_admitted_pixels, ())
        by_eight = analyze_threshold_growth(
            image, (0, 0, 5, 1), (0, 0), 0, 90, threshold_increment=80
        )
        self.assertEqual([frame.threshold for frame in by_eight.frames], [0, 80, 90])
        self.assertEqual(by_eight.frames[1].newly_admitted_pixels, ((1, 0), (2, 0), (3, 0)))
        self.assertEqual(by_eight.frames[1].newly_admitted_count, 3)

    def test_centroid_bounding_rectangle_and_roi_coordinates(self):
        image = Image.new("L", (6, 5), 255)
        image.putpixel((2, 1), 0)
        image.putpixel((3, 1), 0)
        image.putpixel((2, 2), 0)

        result = analyze_threshold_growth(image, (1, 1, 5, 4), (2, 1), 0, 0)
        frame = result.frames[0]

        self.assertEqual(result.roi, (1, 1, 5, 4))
        self.assertEqual(frame.bounding_rectangle, (2, 1, 2, 2))
        self.assertEqual(frame.centroid, (7 / 3, 4 / 3))
        self.assertEqual(frame.pixel_count, 3)
        self.assertEqual(frame.roi_pixel_count, 12)

    def test_sampled_thresholds_include_exact_maximum(self):
        image = Image.new("L", (2, 1))
        image.putdata([0, 100])
        result = analyze_threshold_growth(
            image, (0, 0, 2, 1), (0, 0), 0, 100, threshold_increment=30
        )
        self.assertEqual([frame.threshold for frame in result.frames], [0, 30, 60, 90, 100])
        self.assertEqual(result.frames[-1].newly_admitted_pixels, ((1, 0),))

    def test_background_saturation_is_flagged_once(self):
        image = Image.new("L", (3, 2), 200)
        image.putpixel((0, 0), 0)

        result = analyze_threshold_growth(
            image, (0, 0, 3, 2), (0, 0), 0, 255,
            threshold_increment=100, occupancy_limit_percentage=50,
        )

        self.assertFalse(result.frames[0].background_dominated)
        self.assertTrue(result.frames[-1].background_dominated)
        self.assertEqual(
            sum(
                "background-dominated" in event
                for frame in result.frames for event in frame.events
            ),
            1,
        )

    def test_roi_excludes_outside_pixels_and_seed_must_be_inside(self):
        image = Image.new("L", (4, 2), 255)
        image.putpixel((0, 0), 0)
        image.putpixel((1, 0), 0)
        image.putpixel((2, 0), 0)

        result = analyze_threshold_growth(image, (1, 0, 3, 2), (1, 0), 0, 0)
        self.assertEqual(result.frames[0].component_pixels, ((1, 0), (2, 0)))
        with self.assertRaises(ValueError):
            analyze_threshold_growth(image, (1, 0, 3, 2), (0, 0), 0, 0)

    def test_repeatability_and_original_pixels_are_preserved(self):
        image = Image.new("L", (5, 2), 255)
        image.putdata([0, 20, 255, 0, 255, 255, 40, 255, 255, 255])
        original = image.tobytes()
        args = (image, (0, 0, 5, 2), (0, 0), 0, 40, 3, 75)

        first = analyze_threshold_growth(*args)
        second = analyze_threshold_growth(*args)

        self.assertEqual(first, second)
        self.assertEqual(image.tobytes(), original)

    def test_overlay_colors_growth_and_does_not_change_source(self):
        image = Image.new("L", (3, 3), 0)
        image.putpixel((1, 1), 20)
        rgb = Image.new("RGB", image.size, (25, 30, 35))
        original = rgb.tobytes()
        frame = analyze_threshold_growth(
            image, (0, 0, 3, 3), (0, 0), 0, 20
        ).frames[-1]

        overlay = create_growth_overlay(rgb, frame)

        self.assertEqual(overlay.getpixel((0, 0)), GROWTH_BOUNDARY_COLOR)
        self.assertEqual(overlay.getpixel((1, 1)), GROWTH_NEW_PIXELS_COLOR)
        self.assertEqual(rgb.tobytes(), original)

    def test_component_interior_uses_growth_color(self):
        image = Image.new("L", (3, 3), 0)
        rgb = Image.new("RGB", image.size, (10, 20, 30))
        frame = analyze_threshold_growth(
            image, (0, 0, 3, 3), (1, 1), 0, 0
        ).frames[0]

        overlay = create_growth_overlay(rgb, frame)

        self.assertEqual(overlay.getpixel((1, 1)), GROWTH_COMPONENT_COLOR)


if __name__ == "__main__":
    unittest.main()
