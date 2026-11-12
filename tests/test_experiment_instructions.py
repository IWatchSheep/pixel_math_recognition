import re
import unittest

from PIL import Image

from experiment_instructions import (
    REQUIRED_KEYS,
    parse_experiment_config,
    run_template_auto_align_experiment,
)
from template_matching import auto_align_offset, load_template_variant


EXAMPLE = """\
experiment=template_auto_align
roi_left=1
roi_top=1
roi_right=9
roi_bottom=10
brightness_threshold=160
digit=0
template=thin_zero_independent_xy
uniform_scale=1.00
width_scale=0.90
height_scale=0.85
offset_x=0
offset_y=0
auto_align_radius=2
"""


class ExperimentInstructionTests(unittest.TestCase):
    def test_valid_config_parses_all_fields(self):
        config = parse_experiment_config(EXAMPLE, (12, 12))

        self.assertEqual(config.experiment, "template_auto_align")
        self.assertEqual(config.roi, (1, 1, 9, 10))
        self.assertEqual(config.brightness_threshold, 160)
        self.assertEqual(config.digit, "0")
        self.assertEqual(config.uniform_scale, 1.0)
        self.assertEqual(config.width_scale, 0.9)
        self.assertEqual(config.height_scale, 0.85)
        self.assertEqual(config.auto_align_radius, 2)

    def test_every_required_key_must_be_present(self):
        incomplete = "\n".join(
            line for line in EXAMPLE.splitlines()
            if not line.startswith("height_scale=")
        )

        with self.assertRaisesRegex(ValueError, "height_scale"):
            parse_experiment_config(incomplete, (12, 12))

    def test_unknown_and_duplicate_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown key"):
            parse_experiment_config(EXAMPLE + "surprise=1\n", (12, 12))
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            parse_experiment_config(EXAMPLE + "digit=0\n", (12, 12))

    def test_invalid_numeric_values_and_ranges_are_rejected(self):
        invalid_inputs = (
            EXAMPLE.replace("roi_left=1", "roi_left=1.5"),
            EXAMPLE.replace("uniform_scale=1.00", "uniform_scale=nan"),
            EXAMPLE.replace("width_scale=0.90", "width_scale=1.21"),
            EXAMPLE.replace("height_scale=0.85", "height_scale=abc"),
            EXAMPLE.replace("brightness_threshold=160", "brightness_threshold=256"),
            EXAMPLE.replace("auto_align_radius=2", "auto_align_radius=51"),
            EXAMPLE.replace("digit=0", "digit=10"),
            EXAMPLE.replace("experiment=template_auto_align", "experiment=other"),
        )
        for text in invalid_inputs:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_experiment_config(text, (12, 12))

    def test_out_of_bounds_or_empty_roi_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            parse_experiment_config(EXAMPLE.replace("roi_right=9", "roi_right=13"), (12, 12))
        with self.assertRaisesRegex(ValueError, "positive width"):
            parse_experiment_config(
                EXAMPLE.replace("roi_right=9", "roi_right=1"), (12, 12)
            )

    def test_only_supported_uniform_and_independent_scale_limits_are_accepted(self):
        for key, value in (
            ("uniform_scale", "0.24"),
            ("uniform_scale", "2.51"),
            ("width_scale", "0.79"),
            ("height_scale", "1.21"),
        ):
            with self.subTest(key=key, value=value):
                text = re.sub(
                    rf"^{key}=.*$",
                    f"{key}={value}",
                    EXAMPLE,
                    flags=re.MULTILINE,
                )
                with self.assertRaises(ValueError):
                    parse_experiment_config(text, (12, 12))

    def test_execution_is_deterministic_and_matches_manual_auto_align(self):
        config = parse_experiment_config(EXAMPLE, (12, 12))
        rgb = Image.new("RGB", (12, 12), (255, 255, 255))
        grayscale = Image.new("L", (12, 12), 255)
        for point in ((3, 4), (4, 4), (5, 4), (3, 5), (5, 5), (3, 6), (4, 6), (5, 6)):
            grayscale.putpixel(point, 0)
        rgb_before = rgb.tobytes()
        gray_before = grayscale.tobytes()

        first = run_template_auto_align_experiment(config, rgb, grayscale)
        second = run_template_auto_align_experiment(config, rgb, grayscale)
        roi_mask = first.binary_mask.crop(config.roi)
        template = load_template_variant(config.template_variant, config.digit)
        base_scale = min(
            roi_mask.width / template.width, roi_mask.height / template.height
        )
        manual = auto_align_offset(
            roi_mask,
            template,
            (config.offset_x, config.offset_y),
            config.auto_align_radius,
            base_scale * config.uniform_scale * config.width_scale,
            base_scale * config.uniform_scale * config.height_scale,
        )

        self.assertEqual(first.alignment.selected_offset, manual.selected_offset)
        self.assertEqual(first.alignment.selected_iou, manual.selected_iou)
        self.assertEqual(first.initial_comparison.iou, second.initial_comparison.iou)
        self.assertEqual(first.alignment.selected_offset, second.alignment.selected_offset)
        self.assertEqual(first.overlay.tobytes(), second.overlay.tobytes())
        self.assertEqual(rgb.tobytes(), rgb_before)
        self.assertEqual(grayscale.tobytes(), gray_before)

    def test_invalid_configuration_does_not_change_previous_result_or_source(self):
        source = Image.new("RGB", (12, 12), (22, 33, 44))
        original_pixels = source.tobytes()
        previous_config = parse_experiment_config(EXAMPLE, source.size)
        previous_result = run_template_auto_align_experiment(
            previous_config, source, source.convert("L")
        )
        invalid = EXAMPLE + "unapproved=value\n"

        with self.assertRaises(ValueError):
            parse_experiment_config(invalid, source.size)

        self.assertIs(previous_result.config, previous_config)
        self.assertEqual(previous_result.config, previous_config)
        self.assertEqual(source.tobytes(), original_pixels)

    def test_required_key_set_matches_example_configuration(self):
        keys = {
            line.split("=", 1)[0]
            for line in EXAMPLE.splitlines()
            if "=" in line
        }
        self.assertEqual(keys, REQUIRED_KEYS)


if __name__ == "__main__":
    unittest.main()
