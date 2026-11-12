import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from attachment_aware_fitting import (
    EXPERIMENT_NAME,
    _attachment_loss_fraction,
    _select_attachment_aware_transform,
    _transform_bound_hits,
    parse_attachment_aware_fit_config,
    record_attachment_aware_fit,
    run_attachment_aware_fit,
)
from progressive_anchored_fitting import (
    _render_template_mask,
    _select_transform,
    _template_mask,
    attachment_changes,
    parse_progressive_anchored_fit_config,
    run_progressive_anchored_fit,
)
from template_matching import DIGIT_TEMPLATE_ROWS


def configuration(
        *,
        roi=(0, 0, 9, 9),
        seed=(4, 4),
        thresholds=(0, 1, 1),
        candidates="0,6",
        weights="0,0.02,0.05,0.10",
        auto_record="false",
):
    left, top, right, bottom = roi
    start, maximum, increment = thresholds
    return "\n".join((
        "experiment=progressive_anchored_fit_b",
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
        "cumulative_scale=0.2",
        f"attachment_loss_weights={weights}",
        f"auto_record={auto_record}",
        f"experiment_name={EXPERIMENT_NAME}",
    ))


class AttachmentAwareFittingTests(unittest.TestCase):
    def setUp(self):
        self.rgb = Image.new("RGB", (9, 9), (255, 255, 255))
        self.gray = Image.new("L", (9, 9), 255)
        self.rgb.putpixel((4, 4), (0, 0, 0))
        self.gray.putpixel((4, 4), 0)
        self.config = parse_attachment_aware_fit_config(
            configuration(), self.rgb.size
        )

    def test_parser_requires_zero_and_rejects_bad_or_duplicate_weights(self):
        self.assertEqual(
            self.config.attachment_loss_weights, (0.0, 0.02, 0.05, 0.1)
        )
        for invalid_weights in ("0.02,0.05", "-0.1,0", "0,nan", "0,0.1,0.10"):
            with self.subTest(weights=invalid_weights):
                with self.assertRaises(ValueError):
                    parse_attachment_aware_fit_config(
                        configuration(weights=invalid_weights), self.rgb.size
                    )

    def test_zero_weight_reproduces_stage_1g_progressive_history(self):
        base_config = parse_progressive_anchored_fit_config(
            configuration().replace(
                "experiment=progressive_anchored_fit_b",
                "experiment=progressive_anchored_fit",
            ).replace(
                f"experiment_name={EXPERIMENT_NAME}",
                "experiment_name=stage_1g_baseline",
            ).replace(
                "attachment_loss_weights=0,0.02,0.05,0.10\n", ""
            ),
            self.rgb.size,
        )
        baseline = run_progressive_anchored_fit(
            base_config, self.rgb, self.gray
        )
        weighted = run_attachment_aware_fit(
            parse_attachment_aware_fit_config(
                configuration(weights="0"), self.rgb.size
            ),
            self.rgb,
            self.gray,
        )
        for digit in base_config.candidate_digits:
            old_frames = baseline.frames_for(digit, "progressive")
            new_frames = weighted.frames_for(digit, 0.0)
            self.assertEqual(len(old_frames), len(new_frames))
            for old, new in zip(old_frames, new_frames):
                self.assertEqual(
                    (
                        old.offset_x, old.offset_y, old.rotation_degrees,
                        old.width_scale, old.height_scale,
                    ),
                    new.transform,
                )
                self.assertEqual(old.iou, new.iou)
                self.assertEqual(old.objective, new.objective)
                self.assertEqual(old.matched_pixels, new.matched_pixels)
                self.assertEqual(old.retained_attachments, new.retained_attachments)
                self.assertEqual(old.gained_attachments, new.gained_attachments)
                self.assertEqual(old.lost_attachments, new.lost_attachments)

    def test_every_weight_track_starts_at_the_same_transform(self):
        result = run_attachment_aware_fit(
            self.config, self.rgb, self.gray
        )
        initial = (0, 0, 0, 1.0, 1.0)
        for digit in self.config.base_config.candidate_digits:
            for weight in self.config.attachment_loss_weights:
                self.assertEqual(
                    result.frames_for(digit, weight)[0].transform, initial
                )

    def test_attachment_sets_are_exact_original_coordinates(self):
        previous = frozenset({(501, 702), (502, 702), (503, 703)})
        current = frozenset({(502, 702), (503, 703), (504, 703)})
        retained, gained, lost = attachment_changes(previous, current)
        self.assertEqual(retained, frozenset({(502, 702), (503, 703)}))
        self.assertEqual(gained, frozenset({(504, 703)}))
        self.assertEqual(lost, frozenset({(501, 702)}))
        self.assertEqual(_attachment_loss_fraction(previous, current), 1 / 3)

        rgb = Image.new("RGB", (20, 22), "white")
        gray = Image.new("L", (20, 22), 255)
        gray.putpixel((5, 9), 0)
        gray.putpixel((7, 9), 0)
        gray.putpixel((6, 9), 1)
        rgb.putpixel((5, 9), (0, 0, 0))
        config = parse_attachment_aware_fit_config(
            configuration(
                roi=(5, 7, 14, 16),
                seed=(5, 9),
                candidates="0",
                thresholds=(0, 1, 1),
                weights="0",
            ),
            rgb.size,
        )
        result = run_attachment_aware_fit(config, rgb, gray)
        first, second = result.frames_for("0", 0.0)
        self.assertIn((7, 9), first.newly_selected_pixels)
        self.assertIn((6, 9), second.newly_selected_pixels)
        self.assertIn((7, 9), second.newly_connected_pixels)
        self.assertEqual(first.previous_matched_pixels, ())
        self.assertEqual(first.lost_attachments, ())
        self.assertEqual(
            set(second.previous_matched_pixels), set(first.matched_pixels)
        )
        self.assertEqual(
            set(second.retained_attachments),
            set(first.matched_pixels) & set(second.matched_pixels),
        )
        self.assertEqual(
            set(second.gained_attachments),
            set(second.matched_pixels) - set(first.matched_pixels),
        )
        self.assertEqual(
            set(second.lost_attachments),
            set(first.matched_pixels) - set(second.matched_pixels),
        )
        self.assertTrue(all(5 <= x < 14 and 7 <= y < 16
                            for x, y in second.matched_pixels))

    def test_attachment_penalty_changes_transform_selection(self):
        template = _template_mask("0")
        initial = (0, 0, 0, 1.0, 1.0)
        roi_size = (21, 21)
        config = parse_progressive_anchored_fit_config(
            configuration(
                roi=(0, 0, 21, 21),
                seed=(10, 10),
                candidates="0",
            ).replace(
                "experiment=progressive_anchored_fit_b",
                "experiment=progressive_anchored_fit",
            ).replace(
                f"experiment_name={EXPERIMENT_NAME}",
                "experiment_name=stage_1g_selector_test",
            ).replace("attachment_loss_weights=0,0.02,0.05,0.10\n", ""),
            roi_size,
        )
        prior = _render_template_mask(template, roi_size, initial)
        observed = _render_template_mask(
            template, roi_size, (1, 0, 0, 1.0, 1.0)
        ) | frozenset({(15, 8)})
        baseline = _select_transform(
            observed, template, roi_size, initial, initial, config
        )
        protected = _select_attachment_aware_transform(
            observed,
            template,
            roi_size,
            initial,
            initial,
            config,
            prior,
            100.0,
        )
        self.assertEqual(baseline, (1, 0, 0, 1.0, 1.0))
        self.assertNotEqual(protected, baseline)

    def test_missing_previous_attachments_cause_no_penalty(self):
        template = _template_mask("0")
        initial = (0, 0, 0, 1.0, 1.0)
        roi_size = (21, 21)
        observed = _render_template_mask(
            template, roi_size, (1, 0, 0, 1.0, 1.0)
        ) | frozenset({(15, 8)})
        config = parse_progressive_anchored_fit_config(
            configuration(
                roi=(0, 0, 21, 21), seed=(10, 10), candidates="0"
            ).replace(
                "experiment=progressive_anchored_fit_b",
                "experiment=progressive_anchored_fit",
            ).replace(
                f"experiment_name={EXPERIMENT_NAME}",
                "experiment_name=stage_1g_empty_attachments",
            ).replace("attachment_loss_weights=0,0.02,0.05,0.10\n", ""),
            roi_size,
        )
        self.assertEqual(_attachment_loss_fraction(frozenset(), observed), 0.0)
        self.assertEqual(
            _select_attachment_aware_transform(
                observed, template, roi_size, initial, initial, config,
                frozenset(), 0.1,
            ),
            _select_transform(observed, template, roi_size, initial, initial, config),
        )

    def test_attachment_coordinates_are_not_rebased_to_the_roi(self):
        template = _template_mask("0")
        initial = (0, 0, 0, 1.0, 1.0)
        roi_size = (21, 21)
        roi = (10, 20, 31, 41)
        config = parse_progressive_anchored_fit_config(
            configuration(
                roi=roi, seed=(20, 30), candidates="0"
            ).replace(
                "experiment=progressive_anchored_fit_b",
                "experiment=progressive_anchored_fit",
            ).replace(
                f"experiment_name={EXPERIMENT_NAME}",
                "experiment_name=stage_1g_coordinate_test",
            ).replace("attachment_loss_weights=0,0.02,0.05,0.10\n", ""),
            (40, 50),
        )
        prior_local = _render_template_mask(template, roi_size, initial)
        prior_original = frozenset((x + roi[0], y + roi[1])
                                   for x, y in prior_local)
        observed = _render_template_mask(
            template, roi_size, (1, 0, 0, 1.0, 1.0)
        ) | frozenset({(15, 8)})
        correctly_anchored = _select_attachment_aware_transform(
            observed, template, roi_size, initial, initial, config,
            prior_original, 100.0,
        )
        incorrectly_rebased = _select_attachment_aware_transform(
            observed, template, roi_size, initial, initial, config,
            prior_local, 100.0,
        )
        self.assertEqual(correctly_anchored[0], 0)
        self.assertEqual(incorrectly_rebased, (1, 0, 0, 1.0, 1.0))

    def test_all_transform_bounds_and_per_threshold_steps_are_enforced(self):
        result = run_attachment_aware_fit(self.config, self.rgb, self.gray)
        base = self.config.base_config
        initial = (0, 0, 0, 1.0, 1.0)
        for digit in base.candidate_digits:
            for weight in self.config.attachment_loss_weights:
                frames = result.frames_for(digit, weight)
                for frame in frames:
                    self.assertLessEqual(abs(frame.offset_x), 2)
                    self.assertLessEqual(abs(frame.offset_y), 2)
                    self.assertLessEqual(abs(frame.rotation_degrees), 10)
                    self.assertLessEqual(abs(frame.width_scale - 1), 0.2)
                    self.assertLessEqual(abs(frame.height_scale - 1), 0.2)
                for current in frames[1:]:
                    change = current.transform_change
                    if change is None:
                        self.fail("A noninitial threshold must report transform change.")
                    self.assertLessEqual(abs(change[0]), 1)
                    self.assertLessEqual(abs(change[1]), 1)
                    self.assertLessEqual(abs(change[2]), 5)
                    self.assertLessEqual(abs(change[3]), 0.1)
                    self.assertLessEqual(abs(change[4]), 0.1)
        self.assertEqual(
            _transform_bound_hits(
                (2, -2, 10, 1.2, 0.8),
                initial,
                base,
                (1, -1, 5, 0.1, -0.1),
            ),
            (
                "offset_x_cumulative_max", "offset_x_step",
                "offset_y_cumulative_min", "offset_y_step",
                "rotation_cumulative_max", "rotation_step",
                "width_scale_cumulative_max", "width_scale_step",
                "height_scale_cumulative_min", "height_scale_step",
            ),
        )

    def test_identical_inputs_are_deterministic_and_do_not_mutate_sources(self):
        rgb_before = self.rgb.tobytes()
        gray_before = self.gray.tobytes()
        templates_before = dict(DIGIT_TEMPLATE_ROWS)
        first = run_attachment_aware_fit(self.config, self.rgb, self.gray)
        second = run_attachment_aware_fit(self.config, self.rgb, self.gray)
        summarize = lambda result: tuple(
            (
                frame.digit, frame.attachment_loss_weight, frame.threshold,
                frame.transform, frame.iou, frame.objective,
                frame.previous_matched_pixels, frame.matched_pixels,
                frame.retained_attachments, frame.gained_attachments,
                frame.lost_attachments, frame.transform_bound_hits,
            )
            for frame in result.frames
        )
        self.assertEqual(summarize(first), summarize(second))
        self.assertEqual(self.rgb.tobytes(), rgb_before)
        self.assertEqual(self.gray.tobytes(), gray_before)
        self.assertEqual(DIGIT_TEMPLATE_ROWS, templates_before)

    def test_recorder_writes_settings_histories_overlays_and_zip(self):
        config = parse_attachment_aware_fit_config(
            configuration(candidates="0", weights="0,0.1", auto_record="true"),
            self.rgb.size,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = record_attachment_aware_fit(
                config, self.rgb, self.gray, Path(temporary)
            )
            run_dir = result.output_directory
            self.assertEqual(run_dir.parent.name, EXPERIMENT_NAME)
            self.assertEqual(result.overlay_count, 4)
            self.assertTrue((run_dir / "fitting_history.csv").is_file())
            self.assertTrue((run_dir / "growth_measurements.csv").is_file())
            self.assertTrue((run_dir / "weight_comparisons.csv").is_file())
            self.assertTrue((run_dir / "experiment_settings.txt").is_file())
            self.assertEqual(len(list((run_dir / "overlays").glob("*.png"))), 4)
            with (run_dir / "fitting_history.csv").open(
                    newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            for field in (
                "attachment_loss_weight", "previous_match_count",
                "retained_fraction", "cumulative_gained_attachments",
                "cumulative_lost_attachments", "newly_selected_pixels",
                "newly_connected_pixels", "transform_change",
                "transform_bound_hits", "at_transform_bound",
            ):
                self.assertIn(field, rows[0])
            with zipfile.ZipFile(run_dir / "results.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("fitting_history.csv", names)
            self.assertIn("growth_measurements.csv", names)
            self.assertIn("weight_comparisons.csv", names)
            self.assertIn("experiment_settings.txt", names)
            self.assertEqual(
                len([name for name in names if name.startswith("overlays/")]), 4
            )


if __name__ == "__main__":
    unittest.main()
