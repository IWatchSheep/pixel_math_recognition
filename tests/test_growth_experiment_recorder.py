import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from threading import Event

from PIL import Image

from growth_experiment_recorder import (
    RecordingCancelled,
    record_threshold_growth_experiment,
    render_growth_snapshot,
)
from threshold_growth import analyze_threshold_growth
from threshold_growth_instructions import parse_threshold_growth_config


def configuration(extra=""):
    return parse_threshold_growth_config(
        "experiment=threshold_growth\n"
        "roi_left=0\nroi_top=0\nroi_right=7\nroi_bottom=3\n"
        "brightness_threshold=0\nmaximum_threshold=20\n"
        "threshold_increment=1\nbackground_limit_percent=50\n"
        "seed_x=1\nseed_y=1\nauto_record=true\nauto_zoom=true\n"
        "snapshot_mode=on_change\nexperiment_name=zero_01\n"
        + extra,
        (7, 3),
    )


class GrowthExperimentRecorderTests(unittest.TestCase):
    def setUp(self):
        self.rgb = Image.new("RGB", (7, 3), (245, 245, 245))
        self.gray = Image.new("L", (7, 3), 255)
        for x, value in ((1, 0), (2, 15), (3, 20), (4, 0)):
            self.gray.putpixel((x, 1), value)
            self.rgb.putpixel((x, 1), (value, value, value))
        self.original_rgb = self.rgb.tobytes()
        self.original_gray = self.gray.tobytes()
        self.temporary = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _record(self):
        return record_threshold_growth_experiment(
            configuration(),
            self.rgb,
            self.gray,
            self.output_root,
        )

    def test_records_all_measurements_and_packages_only_requested_files(self):
        result = self._record()
        output = result.output_directory
        expected_thresholds = list(range(21))

        with (output / "growth_measurements.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            measurements = list(csv.DictReader(file))
        with (output / "snapshot_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            manifest = list(csv.DictReader(file))

        self.assertEqual([int(row["threshold"]) for row in measurements], expected_thresholds)
        self.assertEqual([int(row["threshold"]) for row in manifest], expected_thresholds)
        self.assertEqual(manifest[0]["snapshot_file"], "threshold_000.png")
        self.assertEqual(manifest[0]["snapshot_saved"], "true")
        self.assertEqual(manifest[-1]["snapshot_file"], "threshold_020.png")
        self.assertEqual(manifest[-1]["snapshot_saved"], "true")
        self.assertIn("absorbed", manifest[-1]["reason"])
        self.assertTrue((output / "threshold_015.png").is_file())
        self.assertTrue((output / "threshold_020.png").is_file())
        self.assertEqual(
            result.snapshot_files,
            tuple(sorted(result.snapshot_files, key=lambda name: int(name[10:13]))),
        )
        with zipfile.ZipFile(output / "results.zip") as archive:
            names = archive.namelist()
        self.assertIn("growth_measurements.csv", names)
        self.assertIn("snapshot_manifest.csv", names)
        self.assertIn("experiment_settings.txt", names)
        self.assertIn("threshold_020.png", names)
        self.assertNotIn("results.zip", names)
        self.assertEqual(self.rgb.tobytes(), self.original_rgb)
        self.assertEqual(self.gray.tobytes(), self.original_gray)

    def test_capture_changes_and_reuses_identical_final_state_without_duplicate_png(self):
        image = Image.new("RGB", (3, 2), (20, 20, 20))
        grayscale = Image.new("L", image.size, 20)
        config = parse_threshold_growth_config(
            "experiment=threshold_growth\nroi_left=0\nroi_top=0\n"
            "roi_right=3\nroi_bottom=2\nbrightness_threshold=20\n"
            "maximum_threshold=22\nthreshold_increment=1\n"
            "background_limit_percent=50\nseed_x=0\nseed_y=0\n"
            "auto_record=true\nexperiment_name=constant\n",
            image.size,
        )

        result = record_threshold_growth_experiment(
            config, image, grayscale, self.output_root
        )

        self.assertEqual(result.snapshot_files, ("threshold_020.png",))
        with (result.output_directory / "snapshot_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            manifest = list(csv.DictReader(file))
        self.assertEqual(manifest[-1]["snapshot_saved"], "false")
        self.assertEqual(manifest[-1]["snapshot_file"], "threshold_020.png")
        self.assertIn("reused prior snapshot", manifest[-1]["reason"])

    def test_connection_event_is_recorded_and_duplicate_states_are_skipped(self):
        result = self._record()
        with (result.output_directory / "snapshot_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            manifest = list(csv.DictReader(file))
        saved = {
            int(row["threshold"]): row
            for row in manifest
            if row["snapshot_saved"] == "true"
        }

        self.assertIn(20, saved)
        self.assertIn("connection event", saved[20]["reason"])
        self.assertIn("starting section absorbed", saved[20]["reason"])
        self.assertTrue(
            any(manifest[index]["snapshot_saved"] == "false" for index in range(1, 20))
        )

    def test_manifest_distinguishes_untracked_roi_changes_from_tracked_growth(self):
        image = Image.new("RGB", (7, 3), (255, 255, 255))
        grayscale = Image.new("L", image.size, 255)
        grayscale.putpixel((1, 1), 0)
        grayscale.putpixel((5, 1), 5)
        image.putpixel((1, 1), (0, 0, 0))
        image.putpixel((5, 1), (5, 5, 5))
        config = parse_threshold_growth_config(
            "experiment=threshold_growth\n"
            "roi_left=0\nroi_top=0\nroi_right=7\nroi_bottom=3\n"
            "brightness_threshold=0\nmaximum_threshold=10\n"
            "threshold_increment=5\nbackground_limit_percent=50\n"
            "seed_x=1\nseed_y=1\nauto_record=true\n"
            "experiment_name=outside_component\n",
            image.size,
        )

        result = record_threshold_growth_experiment(
            config, image, grayscale, self.output_root
        )
        with (result.output_directory / "snapshot_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            manifest = {
                int(row["threshold"]): row for row in csv.DictReader(file)
            }

        changed = manifest[5]
        self.assertEqual(changed["snapshot_saved"], "true")
        self.assertIn("selected pixels outside tracked component changed", changed["reason"])
        self.assertEqual(changed["tracked_pixel_count"], "1")
        self.assertEqual(changed["selected_outside_tracked_count"], "1")
        self.assertTrue(
            (result.output_directory / changed["snapshot_file"]).is_file()
        )

    def test_snapshot_is_fixed_size_and_uses_original_roi_pixels(self):
        frame = analyze_threshold_growth(
            self.gray, (1, 1, 6, 2), (1, 1), 0, 20
        ).frames[-1]
        first = render_growth_snapshot(self.rgb, frame, (1, 1, 6, 2))
        second = render_growth_snapshot(self.rgb, frame, (1, 1, 6, 2))

        self.assertEqual(first.size, second.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.width, 420)
        self.assertEqual(first.getpixel((4 * 12, 0)), self.rgb.getpixel((5, 1)))

    def test_run_names_do_not_overwrite_prior_results(self):
        first = self._record()
        second = self._record()

        self.assertNotEqual(first.output_directory, second.output_directory)
        self.assertEqual(first.experiment.history, second.experiment.history)
        self.assertEqual(first.snapshot_files, second.snapshot_files)
        self.assertTrue((first.output_directory / "results.zip").is_file())
        self.assertTrue((second.output_directory / "results.zip").is_file())

    def test_cancellation_removes_incomplete_run_directory(self):
        cancellation = Event()

        def cancel_after_first_snapshot(done, total, message):
            if message.startswith("Rendered threshold"):
                cancellation.set()

        with self.assertRaises(RecordingCancelled):
            record_threshold_growth_experiment(
                configuration(),
                self.rgb,
                self.gray,
                self.output_root,
                cancel_event=cancellation,
                progress_callback=cancel_after_first_snapshot,
            )
        parent = self.output_root / "zero_01"
        self.assertFalse(parent.exists() and any(parent.iterdir()))

    def test_pre_cancelled_analysis_writes_nothing(self):
        cancellation = Event()
        cancellation.set()

        with self.assertRaises(RecordingCancelled):
            record_threshold_growth_experiment(
                configuration(),
                self.rgb,
                self.gray,
                self.output_root,
                cancel_event=cancellation,
            )
        parent = self.output_root / "zero_01"
        self.assertFalse(parent.exists() and any(parent.iterdir()))

    def test_file_generation_error_cleans_up_run_directory(self):
        def fail_after_snapshot(done, total, message):
            if message.startswith("Rendered threshold"):
                raise OSError("simulated disk write failure")

        with self.assertRaisesRegex(OSError, "simulated"):
            record_threshold_growth_experiment(
                configuration(),
                self.rgb,
                self.gray,
                self.output_root,
                progress_callback=fail_after_snapshot,
            )
        parent = self.output_root / "zero_01"
        self.assertFalse(parent.exists() and any(parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
