import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from progressive_anchored_fitting import (
    _render_template_mask,
    _select_transform,
    _template_mask,
    attachment_changes,
    parse_progressive_anchored_fit_config,
    record_progressive_anchored_fit,
    run_progressive_anchored_fit,
)


def configuration(
        *,
        roi=(0, 0, 9, 9),
        seed=(4, 4),
        thresholds=(80, 160, 80),
        candidates="0,6,8,9",
        auto_record="false",
):
    left, top, right, bottom = roi
    start, maximum, increment = thresholds
    return "\n".join((
        "experiment=progressive_anchored_fit",
        f"roi_left={left}", f"roi_top={top}",
        f"roi_right={right}", f"roi_bottom={bottom}",
        f"seed_x={seed[0]}", f"seed_y={seed[1]}",
        f"brightness_threshold={start}", f"maximum_threshold={maximum}",
        f"threshold_increment={increment}",
        "background_limit_percent=100",
        f"candidate_digits={candidates}",
        "initial_offset_x=0", "initial_offset_y=0",
        "initial_rotation_degrees=0", "initial_width_scale=1.0",
        "initial_height_scale=1.0", "step_translation_pixels=1",
        "cumulative_translation_pixels=2", "step_rotation_degrees=5",
        "cumulative_rotation_degrees=10", "step_scale=0.1",
        "cumulative_scale=0.2", f"auto_record={auto_record}",
        "experiment_name=stage_1g_test",
    ))


class ProgressiveAnchoredFittingTests(unittest.TestCase):
    def setUp(self):
        self.rgb = Image.new("RGB", (9, 9), "white")
        self.gray = Image.new("L", (9, 9), 255)
        self.rgb.putpixel((4, 4), (0, 0, 0))
        self.gray.putpixel((4, 4), 0)
        self.config = parse_progressive_anchored_fit_config(
            configuration(), self.rgb.size
        )

    def test_parser_accepts_complete_config_and_rejects_bad_limits(self):
        self.assertEqual(self.config.candidate_digits, ("0", "6", "8", "9"))
        invalid = configuration().replace(
            "cumulative_rotation_degrees=10", "cumulative_rotation_degrees=4"
        )
        with self.assertRaisesRegex(ValueError, "at least the per-step limit"):
            parse_progressive_anchored_fit_config(invalid, self.rgb.size)

    def test_fixed_and_progressive_start_from_identical_transform(self):
        result = run_progressive_anchored_fit(self.config, self.rgb, self.gray)
        for digit in self.config.candidate_digits:
            fixed = result.frames_for(digit, "fixed")[0]
            progressive = result.frames_for(digit, "progressive")[0]
            self.assertEqual(
                (fixed.offset_x, fixed.offset_y, fixed.rotation_degrees,
                 fixed.width_scale, fixed.height_scale),
                (progressive.offset_x, progressive.offset_y,
                 progressive.rotation_degrees, progressive.width_scale,
                 progressive.height_scale),
            )
            self.assertEqual((fixed.offset_x, fixed.offset_y), (0, 0))

    def test_rotation_and_independent_scaling_change_rasterized_template(self):
        template = _template_mask("0")
        base = _render_template_mask(template, (21, 21), (0, 0, 0, 1.0, 1.0))
        rotated = _render_template_mask(template, (21, 21), (0, 0, 25, 1.0, 1.0))
        wide = _render_template_mask(template, (21, 21), (0, 0, 0, 1.2, 1.0))
        tall = _render_template_mask(template, (21, 21), (0, 0, 0, 1.0, 1.2))
        self.assertNotEqual(base, rotated)
        self.assertNotEqual(base, wide)
        self.assertNotEqual(base, tall)

    def test_progressive_transforms_remain_within_cumulative_and_step_limits(self):
        result = run_progressive_anchored_fit(self.config, self.rgb, self.gray)
        for digit in self.config.candidate_digits:
            frames = result.frames_for(digit, "progressive")
            for frame in frames:
                self.assertLessEqual(abs(frame.offset_x), 2)
                self.assertLessEqual(abs(frame.offset_y), 2)
                self.assertLessEqual(abs(frame.rotation_degrees), 10)
                self.assertLessEqual(abs(frame.width_scale - 1.0), 0.2)
                self.assertLessEqual(abs(frame.height_scale - 1.0), 0.2)
            for previous, current in zip(frames, frames[1:]):
                self.assertLessEqual(abs(current.offset_x - previous.offset_x), 1)
                self.assertLessEqual(abs(current.offset_y - previous.offset_y), 1)
                self.assertLessEqual(
                    abs(current.rotation_degrees - previous.rotation_degrees), 5
                )
                self.assertLessEqual(abs(current.width_scale - previous.width_scale), 0.1)
                self.assertLessEqual(abs(current.height_scale - previous.height_scale), 0.1)

    def test_select_transform_uses_deterministic_bounded_search(self):
        template = _template_mask("0")
        initial = (0, 0, 0, 1.0, 1.0)
        observed = _render_template_mask(template, (21, 21), (1, 0, 0, 1.0, 1.0))
        config = parse_progressive_anchored_fit_config(
            configuration(roi=(0, 0, 21, 21), seed=(10, 10)), (21, 21)
        )
        selected = _select_transform(
            observed, template, (21, 21), initial, initial, config
        )
        self.assertEqual(
            selected,
            _select_transform(observed, template, (21, 21), initial, initial, config),
        )
        self.assertEqual(selected, (1, 0, 0, 1.0, 1.0))
        self.assertLessEqual(abs(selected[0] - initial[0]), 1)
        self.assertLessEqual(abs(selected[1] - initial[1]), 1)
        self.assertLessEqual(abs(selected[2] - initial[2]), 5)
        self.assertLessEqual(abs(selected[3] - initial[3]), 0.1)
        self.assertLessEqual(abs(selected[4] - initial[4]), 0.1)
        rotated_and_scaled = _render_template_mask(
            template, (21, 21), (0, 0, 5, 1.1, 0.9)
        )
        self.assertEqual(
            _select_transform(
                rotated_and_scaled, template, (21, 21), initial, initial, config
            ),
            (0, 0, 5, 1.1, 0.9),
        )

    def test_retained_gained_lost_and_reappearing_attachments(self):
        first = frozenset({(1, 1), (2, 2)})
        second = frozenset({(2, 2), (3, 3)})
        retained, gained, lost = attachment_changes(first, second)
        self.assertEqual(retained, frozenset({(2, 2)}))
        self.assertEqual(gained, frozenset({(3, 3)}))
        self.assertEqual(lost, frozenset({(1, 1)}))
        retained, gained, lost = attachment_changes(second, first)
        self.assertEqual(retained, frozenset({(2, 2)}))
        self.assertEqual(gained, frozenset({(1, 1)}))
        self.assertEqual(lost, frozenset({(3, 3)}))

    def test_newly_selected_and_newly_connected_pixels_are_separate(self):
        rgb = Image.new("RGB", (7, 5), "white")
        gray = Image.new("L", (7, 5), 255)
        for x, value in ((1, 0), (3, 70), (2, 120)):
            gray.putpixel((x, 2), value)
            rgb.putpixel((x, 2), (value, value, value))
        config = parse_progressive_anchored_fit_config(
            configuration(roi=(0, 0, 7, 5), seed=(1, 2), candidates="0"),
            rgb.size,
        )
        result = run_progressive_anchored_fit(config, rgb, gray)
        at_80 = result.frames_for("0", "fixed")[0]
        at_160 = result.frames_for("0", "fixed")[1]
        self.assertEqual(at_80.newly_selected_pixels, ((1, 2), (3, 2)))
        self.assertEqual(at_80.newly_connected_pixels, ())
        self.assertEqual(at_160.newly_selected_pixels, ((2, 2),))
        self.assertIn((3, 2), at_160.newly_connected_pixels)
        self.assertNotIn((2, 2), at_160.newly_connected_pixels)

    def test_candidate_comparisons_are_deterministic_and_preserve_source(self):
        original_rgb = self.rgb.copy()
        original_gray = self.gray.copy()
        first = run_progressive_anchored_fit(self.config, self.rgb, self.gray)
        second = run_progressive_anchored_fit(self.config, self.rgb, self.gray)
        summarize = lambda result: tuple(
            (frame.digit, frame.method, frame.threshold, frame.offset_x,
             frame.offset_y, frame.rotation_degrees, frame.width_scale,
             frame.height_scale, frame.matched_pixels,
             frame.retained_attachments, frame.gained_attachments,
             frame.lost_attachments)
            for frame in result.frames
        )
        self.assertEqual(summarize(first), summarize(second))
        self.assertEqual(self.rgb.tobytes(), original_rgb.tobytes())
        self.assertEqual(self.gray.tobytes(), original_gray.tobytes())

    def test_recorder_writes_csv_overlays_and_zip_archive(self):
        config = parse_progressive_anchored_fit_config(
            configuration(candidates="0", auto_record="true"), self.rgb.size
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = record_progressive_anchored_fit(
                config, self.rgb, self.gray, Path(temporary)
            )
            run_dir = result.output_directory
            self.assertEqual(result.overlay_count, 4)
            self.assertTrue((run_dir / "fitting_history.csv").is_file())
            self.assertTrue((run_dir / "growth_measurements.csv").is_file())
            self.assertTrue((run_dir / "fixed_vs_progressive.csv").is_file())
            self.assertEqual(len(list((run_dir / "overlays").glob("*.png"))), 4)
            with (run_dir / "fitting_history.csv").open(
                    newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertIn("cumulative_offset_x", rows[0])
            self.assertIn("cumulative_rotation_degrees", rows[0])
            self.assertIn("cumulative_width_scale_delta", rows[0])
            with zipfile.ZipFile(run_dir / "results.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("fitting_history.csv", names)
            self.assertIn("growth_measurements.csv", names)
            self.assertIn("fixed_vs_progressive.csv", names)
            self.assertIn("experiment_settings.txt", names)
            self.assertEqual(
                len([name for name in names if name.startswith("overlays/")]), 4
            )


if __name__ == "__main__":
    unittest.main()
