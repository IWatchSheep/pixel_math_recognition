import unittest

from main import next_zoom_level


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


if __name__ == "__main__":
    unittest.main()
