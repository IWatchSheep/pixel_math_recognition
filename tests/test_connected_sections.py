import unittest

from PIL import Image

from connected_sections import analyze_component_merge, find_connected_sections


def mask_from_rows(rows):
    image = Image.new("L", (len(rows[0]), len(rows)), 255)
    image.putdata([0 if value == "1" else 255 for row in rows for value in row])
    return image


def grayscale_from_rows(rows):
    image = Image.new("L", (len(rows[0].split()), len(rows)), 255)
    image.putdata([int(value) for row in rows for value in row.split()])
    return image


class ConnectedSectionTests(unittest.TestCase):
    def test_finds_independent_sections_and_bounding_rectangles(self):
        mask = mask_from_rows(
            [
                "11000",
                "11010",
                "00010",
            ]
        )

        sections, labels = find_connected_sections(mask)

        self.assertEqual([section.pixel_count for section in sections], [4, 2])
        self.assertEqual(sections[0].bounding_rectangle, (0, 0, 2, 2))
        self.assertEqual(sections[1].bounding_rectangle, (3, 1, 1, 2))
        self.assertEqual(len(labels), 15)

    def test_diagonal_pixels_connect_with_eight_neighbor_connectivity(self):
        mask = mask_from_rows(["100", "010", "001"])

        sections, _labels = find_connected_sections(mask)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].pixel_count, 3)
        self.assertEqual(sections[0].bounding_rectangle, (0, 0, 3, 3))

    def test_results_are_consistent_when_repeated(self):
        mask = mask_from_rows(["1100", "0110", "0001"])

        first_sections, first_labels = find_connected_sections(mask)
        second_sections, second_labels = find_connected_sections(mask)

        self.assertEqual(first_sections, second_sections)
        self.assertEqual(first_labels, second_labels)

    def test_white_image_has_no_sections(self):
        sections, labels = find_connected_sections(Image.new("L", (2, 2), 255))

        self.assertEqual(sections, [])
        self.assertEqual(list(labels), [0, 0, 0, 0])

    def test_merge_reports_exact_faint_bridge_and_new_pixels_separately(self):
        image = grayscale_from_rows(
            [
                "255 255 255 255 100",
                "0 80 100 90 0",
                "255 255 255 255 255",
            ]
        )

        result = analyze_component_merge(image, 50, (0, 1), (4, 1))

        self.assertEqual(result.merge_threshold, 100)
        self.assertEqual(result.seed_section_sizes, (1, 1))
        self.assertEqual(result.newly_selected_pixels, ((4, 0), (2, 1)))
        self.assertEqual(result.connecting_path, tuple((x, 1) for x in range(5)))
        self.assertEqual(result.bridge_pixels, ((1, 1), (2, 1), (3, 1)))
        self.assertEqual(result.merged_component_size, 6)
        self.assertEqual(result.absorbed_start_sections, 0)
        self.assertEqual(result.selected_pixel_count, 6)
        self.assertEqual(result.roi_pixel_count, 15)

    def test_already_connected_seeds_have_start_threshold_as_merge(self):
        image = grayscale_from_rows(["0 0 0"])

        result = analyze_component_merge(image, 10, (0, 0), (2, 0))

        self.assertEqual(result.merge_threshold, 10)
        self.assertEqual(result.newly_selected_pixels, ())
        self.assertEqual(result.bridge_pixels, ())
        self.assertEqual(result.merged_component_size, 3)

    def test_sections_that_do_not_connect_in_range_return_no_merge(self):
        image = grayscale_from_rows(["0 255 0"])

        result = analyze_component_merge(image, 10, (0, 0), (2, 0), max_threshold=200)

        self.assertIsNone(result.merge_threshold)
        self.assertEqual(result.connecting_path, ())
        self.assertEqual(result.merged_component_size, 0)

    def test_equal_brightness_pixels_are_processed_deterministically(self):
        image = grayscale_from_rows(["0 100 100 0"])

        first = analyze_component_merge(image, 20, (0, 0), (3, 0))
        second = analyze_component_merge(image, 20, (0, 0), (3, 0))

        self.assertEqual(first, second)
        self.assertEqual(first.merge_threshold, 100)
        self.assertEqual(first.newly_selected_pixels, ((1, 0), (2, 0)))
        self.assertEqual(first.bridge_pixels, ((1, 0), (2, 0)))

    def test_merge_through_another_starting_section_reports_absorption(self):
        image = grayscale_from_rows(["0 0 80 0 80 0 0"])

        result = analyze_component_merge(image, 10, (0, 0), (6, 0))

        self.assertEqual(result.merge_threshold, 80)
        self.assertEqual(result.merged_component_size, 7)
        self.assertEqual(result.absorbed_start_sections, 1)

    def test_roi_excludes_an_external_connecting_path(self):
        image = grayscale_from_rows(
            [
                "80 80 80 80 80",
                "0 255 255 255 0",
                "255 255 255 255 255",
            ]
        )

        result = analyze_component_merge(
            image, 10, (0, 1), (4, 1), max_threshold=100, roi=(0, 1, 5, 3)
        )

        self.assertIsNone(result.merge_threshold)
        self.assertEqual(result.roi, (0, 1, 5, 3))

    def test_high_brightness_background_connection_is_reported_at_high_threshold(self):
        image = grayscale_from_rows(["0 190 190 0"])

        result = analyze_component_merge(image, 10, (0, 0), (3, 0))

        self.assertEqual(result.merge_threshold, 190)
        self.assertEqual(result.newly_selected_pixels, ((1, 0), (2, 0)))
        self.assertEqual(result.bridge_pixels, ((1, 0), (2, 0)))

    def test_rejects_seed_outside_roi_or_not_selected_at_start(self):
        image = grayscale_from_rows(["0 255 0"])

        with self.assertRaises(ValueError):
            analyze_component_merge(image, 10, (0, 0), (2, 0), roi=(0, 0, 2, 1))
        with self.assertRaises(ValueError):
            analyze_component_merge(image, 10, (1, 0), (2, 0))


if __name__ == "__main__":
    unittest.main()
