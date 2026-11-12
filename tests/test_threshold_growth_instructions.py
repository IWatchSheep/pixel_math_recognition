import unittest

from PIL import Image

from threshold_growth_instructions import (
    parse_threshold_growth_config,
    run_threshold_growth_experiment,
)


BASE_CONFIG = """\
experiment=threshold_growth
roi_left=1
roi_top=1
roi_right=7
roi_bottom=6
brightness_threshold=80
maximum_threshold=160
threshold_increment=40
background_limit_percent=50
"""


class ThresholdGrowthInstructionsTests(unittest.TestCase):
    def setUp(self):
        self.rgb = Image.new("RGB", (8, 7), "white")
        self.gray = Image.new("L", self.rgb.size, 255)
        for point in ((2, 2), (3, 2), (5, 4)):
            self.gray.putpixel(point, 40)
            self.rgb.putpixel(point, (40, 40, 40))
        self.gray.putpixel((4, 3), 100)

    def test_parses_original_image_roi_and_optional_seed(self):
        config = parse_threshold_growth_config(
            BASE_CONFIG
            + "seed_x=2\nseed_y=2\nauto_record=true\nauto_zoom=true\n"
            "snapshot_mode=on_change\nexperiment_name=zero_01\n",
            self.rgb.size,
        )

        self.assertEqual(config.roi, (1, 1, 7, 6))
        self.assertEqual(config.seed, (2, 2))
        self.assertEqual(config.brightness_threshold, 80)
        self.assertTrue(config.auto_record)
        self.assertTrue(config.auto_zoom)
        self.assertEqual(config.snapshot_mode, "on_change")
        self.assertEqual(config.experiment_name, "zero_01")

    def test_auto_record_and_recording_options_are_strictly_validated(self):
        with self.assertRaisesRegex(ValueError, "requires explicit seed"):
            parse_threshold_growth_config(
                BASE_CONFIG + "auto_record=true\n", self.rgb.size
            )
        for key, value in (
            ("auto_record", "yes"),
            ("auto_zoom", "1"),
            ("snapshot_mode", "every_threshold"),
            ("experiment_name", "../outside"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    parse_threshold_growth_config(
                        BASE_CONFIG + f"{key}={value}\n", self.rgb.size
                    )

    def test_rejects_missing_or_mismatched_seed_coordinates(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            parse_threshold_growth_config(BASE_CONFIG + "seed_x=2\n", self.rgb.size)
        with self.assertRaisesRegex(ValueError, "inside the ROI"):
            parse_threshold_growth_config(
                BASE_CONFIG + "seed_x=0\nseed_y=2\n", self.rgb.size
            )

    def test_rejects_bad_schema_and_invalid_ranges(self):
        invalid_configs = (
            BASE_CONFIG.replace("threshold_growth", "template_auto_align"),
            BASE_CONFIG + "unexpected=yes\n",
            BASE_CONFIG + "brightness_threshold=80\n",
            BASE_CONFIG.replace("roi_right=7", "roi_right=9"),
            BASE_CONFIG.replace("maximum_threshold=160", "maximum_threshold=79"),
            BASE_CONFIG.replace("threshold_increment=40", "threshold_increment=0"),
            BASE_CONFIG.replace("background_limit_percent=50", "background_limit_percent=0"),
        )
        for text in invalid_configs:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_threshold_growth_config(text, self.rgb.size)

    def test_no_seed_returns_components_with_stable_coordinates_and_no_history(self):
        config = parse_threshold_growth_config(BASE_CONFIG, self.rgb.size)
        original_rgb = self.rgb.tobytes()
        original_gray = self.gray.tobytes()

        result = run_threshold_growth_experiment(config, self.rgb, self.gray)

        self.assertEqual(
            [(section.number, section.pixel_count, section.bounding_rectangle)
             for section in result.sections],
            [(1, 2, (2, 2, 2, 1)), (2, 1, (5, 4, 1, 1))],
        )
        self.assertEqual(
            [
                next(
                    (x, y)
                    for y in range(section.min_y, section.max_y + 1)
                    for x in range(section.min_x, section.max_x + 1)
                    if result.section_labels[y * self.rgb.width + x] == section.number
                )
                for section in result.sections
            ],
            [(2, 2), (5, 4)],
        )
        self.assertIsNone(result.history)
        self.assertEqual(self.rgb.tobytes(), original_rgb)
        self.assertEqual(self.gray.tobytes(), original_gray)

    def test_seed_runs_existing_growth_analysis_for_that_component(self):
        config = parse_threshold_growth_config(
            BASE_CONFIG + "seed_x=2\nseed_y=2\n", self.rgb.size
        )

        result = run_threshold_growth_experiment(config, self.rgb, self.gray)

        self.assertEqual(result.history.frames[0].seed, (2, 2))
        self.assertEqual(result.history.frames[0].pixel_count, 2)

    def test_seed_not_selected_at_start_threshold_is_rejected(self):
        config = parse_threshold_growth_config(
            BASE_CONFIG + "seed_x=4\nseed_y=3\n", self.rgb.size
        )

        with self.assertRaisesRegex(ValueError, "not selected"):
            run_threshold_growth_experiment(config, self.rgb, self.gray)


if __name__ == "__main__":
    unittest.main()
