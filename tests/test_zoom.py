import unittest

from main import fit_zoom_to_roi, next_zoom_level


class ZoomLevelTests(unittest.TestCase):
    def test_zoom_out_from_default_steps_through_one_to_one(self):
        zoom = 10.0
        levels = []
        while zoom > 1.0:
            zoom = next_zoom_level(zoom, -1)
            levels.append(zoom)

        self.assertEqual(levels, [8.0, 6.0, 4.0, 2.0, 1.0])

    def test_zoom_in_and_out_are_reversible_at_supported_levels(self):
        for zoom in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 64.0):
            larger = next_zoom_level(zoom, 1)
            if zoom < 64.0:
                self.assertEqual(next_zoom_level(larger, -1), zoom)

    def test_fractional_auto_zoom_moves_to_adjacent_supported_level(self):
        self.assertEqual(next_zoom_level(0.63, -1), 0.5)
        self.assertEqual(next_zoom_level(0.63, 1), 0.75)
        self.assertEqual(next_zoom_level(1.5, -1), 1.0)
        self.assertEqual(next_zoom_level(1.5, 1), 2.0)

    def test_zoom_steps_stop_at_supported_limits(self):
        self.assertEqual(next_zoom_level(0.01, -1), 0.01)
        self.assertEqual(next_zoom_level(64.0, 1), 64.0)

    def test_invalid_zoom_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            next_zoom_level(1.0, 0)

    def test_auto_zoom_uses_available_viewport_without_rounding_down(self):
        roi = (649, 138, 677, 171)
        viewport = (955, 685)

        zoom = fit_zoom_to_roi(roi, viewport)

        self.assertAlmostEqual(zoom, 661 / 33)
        self.assertGreater(zoom, 16.0)
        self.assertLessEqual((roi[2] - roi[0]) * zoom, viewport[0] - 24)
        self.assertAlmostEqual((roi[3] - roi[1]) * zoom, viewport[1] - 24)

    def test_auto_zoom_scales_down_to_fit_and_respects_zoom_cap(self):
        self.assertAlmostEqual(fit_zoom_to_roi((0, 0, 100, 100), (500, 300)), 2.76)
        self.assertEqual(fit_zoom_to_roi((0, 0, 2, 2), (1000, 1000)), 64.0)

    def test_auto_zoom_rejects_empty_roi_or_viewport(self):
        with self.assertRaises(ValueError):
            fit_zoom_to_roi((2, 2, 2, 5), (500, 300))
        with self.assertRaises(ValueError):
            fit_zoom_to_roi((0, 0, 10, 10), (0, 300))


if __name__ == "__main__":
    unittest.main()
