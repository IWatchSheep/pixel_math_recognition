import unittest

from PIL import Image

from template_matching import (
    IMAGE_ONLY_COLOR,
    OVERLAP_COLOR,
    auto_align_offset,
    TEMPLATE_VARIANT_DIGIT,
    TEMPLATE_VARIANT_THIN_ZERO,
    TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
    TEMPLATE_ONLY_COLOR,
    TemplateFitSettings,
    TemplateTransform,
    compare_template,
    create_template_overlay,
    format_roi_coordinates,
    load_digit_template,
    load_template_variant,
    scale_binary_template,
    transfer_template_fit,
)


def binary_mask(rows):
    return Image.frombytes(
        "L",
        (len(rows[0]), len(rows)),
        bytes(0 if value == "1" else 255 for row in rows for value in row),
    )


class TemplateMatchingTests(unittest.TestCase):
    def test_digit_templates_are_available_as_binary_5_by_7_masks(self):
        for digit in map(str, range(10)):
            with self.subTest(digit=digit):
                template = load_digit_template(digit)
                self.assertEqual(template.size, (5, 7))
                self.assertEqual(template.mode, "L")
                self.assertTrue(set(template.tobytes()) <= {0, 255})
                self.assertIn(0, template.tobytes())

    def test_unknown_digit_is_rejected(self):
        with self.assertRaises(ValueError):
            load_digit_template("x")

    def test_existing_digit_zero_template_is_unchanged(self):
        self.assertEqual(
            list(load_digit_template("0").tobytes()),
            [
                255 if pixel == "0" else 0
                for row in (
                    "01110", "10001", "10001", "10001",
                    "10001", "10001", "01110",
                )
                for pixel in row
            ],
        )
        self.assertEqual(
            load_template_variant(TEMPLATE_VARIANT_DIGIT, "0").tobytes(),
            load_digit_template("0").tobytes(),
        )

    def test_thin_zero_variants_use_the_same_manually_defined_outline(self):
        uniform = load_template_variant(TEMPLATE_VARIANT_THIN_ZERO)
        independent = load_template_variant(TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT)

        self.assertEqual(uniform.size, (7, 9))
        self.assertEqual(uniform.tobytes(), independent.tobytes())
        self.assertEqual(uniform.getpixel((3, 0)), 0)
        self.assertEqual(uniform.getpixel((3, 4)), 255)
        self.assertEqual(uniform.getpixel((0, 4)), 0)
        with self.assertRaises(ValueError):
            load_template_variant(TEMPLATE_VARIANT_THIN_ZERO, "2")

    def test_uniform_scaling_uses_nearest_neighbor_and_preserves_binary_values(self):
        source = binary_mask(("10", "01"))
        scaled = scale_binary_template(source, 2)

        self.assertEqual(scaled.size, (4, 4))
        self.assertEqual(
            list(scaled.tobytes()),
            [0, 0, 255, 255, 0, 0, 255, 255,
             255, 255, 0, 0, 255, 255, 0, 0],
        )

    def test_independent_scaling_changes_width_and_height_separately(self):
        source = binary_mask(("10", "01", "11"))

        scaled = scale_binary_template(source, 1.5, 0.5)

        self.assertEqual(scaled.size, (3, 2))
        self.assertEqual(set(scaled.tobytes()), {0, 255})

    def test_independent_scale_multipliers_are_bounded(self):
        transform = TemplateTransform(4.0)
        transform.set_independent_scales(0.8, 1.2)
        self.assertEqual((transform.scale, transform.scale_y), (3.2, 4.8))
        for width, height in ((0.79, 1.0), (1.0, 1.21)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    transform.set_independent_scales(width, height)

    def test_translation_repositions_template_and_exposes_only_selected_pixels(self):
        image_mask = Image.new("L", (5, 1), 255)
        image_mask.putpixel((3, 0), 0)
        template = binary_mask(("10",))

        result = compare_template(image_mask, template, offset_x=2, scale=1)

        self.assertEqual(result.template_mask.getpixel((3, 0)), 0)
        self.assertEqual(result.template_mask.getpixel((2, 0)), 255)
        self.assertEqual(result.iou, 1.0)

    def test_perfect_match_scores_one_hundred_percent(self):
        template = load_digit_template("8")
        image_mask = scale_binary_template(template, 2)

        result = compare_template(image_mask, template, scale=2)

        self.assertEqual(result.iou, 1.0)
        self.assertEqual(result.intersection_pixels, image_mask.tobytes().count(0))
        self.assertEqual(result.image_only_pixels, 0)
        self.assertEqual(result.template_only_pixels, 0)

    def test_no_overlap_scores_zero(self):
        image_mask = binary_mask(("1000000",))
        template = binary_mask(("1",))

        result = compare_template(image_mask, template, offset_x=20)

        self.assertEqual(result.iou, 0.0)
        self.assertEqual(result.image_only_pixels, 1)
        self.assertEqual(result.template_only_pixels, 0)

    def test_partial_overlap_calculates_iou_and_visual_categories(self):
        image_mask = binary_mask(("100",))
        template = binary_mask(("11",))

        result = compare_template(image_mask, template)

        self.assertEqual(result.iou, 0.5)
        self.assertEqual(result.intersection_pixels, 1)
        self.assertEqual(result.image_only_pixels, 0)
        self.assertEqual(result.template_only_pixels, 1)
        self.assertEqual(result.visualization.getpixel((0, 0)), OVERLAP_COLOR)
        self.assertEqual(result.visualization.getpixel((1, 0)), TEMPLATE_ONLY_COLOR)
        self.assertEqual(result.visualization.getpixel((2, 0)), (0, 0, 0))

    def test_anisotropic_comparison_reports_exact_overlap_category_counts(self):
        image_mask = binary_mask(("1100", "0100"))
        template = binary_mask(("10", "11"))

        result = compare_template(image_mask, template, scale=1, scale_y=1)

        self.assertEqual(result.intersection_pixels, 2)
        self.assertEqual(result.image_only_pixels, 1)
        self.assertEqual(result.template_only_pixels, 1)
        self.assertEqual(result.union_pixels, 4)
        self.assertEqual(result.iou, 0.5)
        self.assertEqual(result.scale_y, 1)

    def test_auto_align_recovers_known_integer_offset(self):
        image_mask = Image.new("L", (7, 5), 255)
        image_mask.putpixel((5, 1), 0)
        template = binary_mask(("1",))

        result = auto_align_offset(image_mask, template, (0, 0), 3)

        self.assertEqual(result.starting_offset, (0, 0))
        self.assertEqual(result.selected_offset, (2, -1))
        self.assertEqual(result.starting_iou, 0.0)
        self.assertEqual(result.selected_iou, 1.0)
        self.assertEqual(result.improvement_percentage_points, 100.0)

    def test_auto_align_ties_choose_smallest_displacement_then_y_then_x(self):
        template = binary_mask(("1",))
        equal_distance = Image.new("L", (7, 3), 255)
        equal_distance.putpixel((2, 1), 0)
        equal_distance.putpixel((4, 1), 0)

        result = auto_align_offset(equal_distance, template, (0, 0), 2)

        self.assertEqual(result.selected_offset, (-1, 0))

        equal_scores = Image.new("L", (7, 3), 255)
        equal_scores.putpixel((2, 0), 0)
        equal_scores.putpixel((2, 2), 0)
        tie_result = auto_align_offset(equal_scores, template, (0, 0), 1)

        self.assertEqual(tie_result.selected_offset, (-1, -1))

    def test_auto_align_with_no_improvement_keeps_starting_offset(self):
        image_mask = binary_mask(("100",))
        template = binary_mask(("1",))

        result = auto_align_offset(image_mask, template, (20, 0), 3)

        self.assertEqual(result.selected_offset, result.starting_offset)
        self.assertEqual(result.starting_iou, result.selected_iou)
        self.assertEqual(result.improvement_percentage_points, 0.0)

    def test_auto_align_clips_templates_at_roi_boundaries(self):
        image_mask = Image.new("L", (3, 3), 255)
        image_mask.putpixel((0, 0), 0)
        template = binary_mask(("00", "01"))

        result = auto_align_offset(image_mask, template, (0, 0), 1)

        self.assertEqual(result.selected_offset, (-1, -1))
        self.assertEqual(result.selected_iou, 1.0)
        self.assertEqual(result.comparison.template_mask.size, image_mask.size)
        self.assertEqual(result.comparison.template_mask.getpixel((0, 0)), 0)
        self.assertEqual(sum(value == 0 for value in result.comparison.template_mask.tobytes()), 1)

    def test_auto_align_is_deterministic_and_does_not_mutate_masks(self):
        image_mask = Image.new("L", (9, 7), 255)
        image_mask.putpixel((6, 2), 0)
        template = binary_mask(("1",))
        image_before = image_mask.tobytes()
        template_before = template.tobytes()

        first = auto_align_offset(image_mask, template, (0, 0), 5)
        second = auto_align_offset(image_mask, template, (0, 0), 5)

        self.assertEqual(first.selected_offset, second.selected_offset)
        self.assertEqual(first.selected_iou, second.selected_iou)
        self.assertEqual(first.improvement_percentage_points, second.improvement_percentage_points)
        self.assertEqual(first.comparison.visualization.tobytes(), second.comparison.visualization.tobytes())
        self.assertEqual(image_mask.tobytes(), image_before)
        self.assertEqual(template.tobytes(), template_before)

    def test_auto_align_rejects_out_of_range_search_radius(self):
        with self.assertRaises(ValueError):
            auto_align_offset(binary_mask(("1",)), binary_mask(("1",)), (0, 0), 51)

    def test_image_only_pixels_have_a_distinct_overlay_color(self):
        image_mask = binary_mask(("111",))
        template = binary_mask(("1",))

        result = compare_template(image_mask, template)

        self.assertEqual(result.visualization.getpixel((0, 0)), IMAGE_ONLY_COLOR)
        self.assertEqual(result.visualization.getpixel((1, 0)), OVERLAP_COLOR)
        self.assertEqual(result.visualization.getpixel((2, 0)), IMAGE_ONLY_COLOR)

    def test_overlay_colors_roi_on_copy_and_preserves_original_rgb_pixels(self):
        image_mask = binary_mask(("100",))
        comparison = compare_template(image_mask, binary_mask(("11",)))
        source = Image.new("RGB", (5, 3), (12, 34, 56))
        original_pixels = source.tobytes()

        overlay = create_template_overlay(source, (1, 1, 4, 2), comparison)

        self.assertEqual(source.tobytes(), original_pixels)
        self.assertEqual(overlay.getpixel((1, 1)), OVERLAP_COLOR)
        self.assertEqual(overlay.getpixel((2, 1)), TEMPLATE_ONLY_COLOR)
        self.assertEqual(overlay.getpixel((0, 0)), (12, 34, 56))

    def test_empty_masks_are_undefined_and_one_sided_empty_is_zero(self):
        empty = Image.new("L", (3, 1), 255)
        selected = binary_mask(("100",))

        both_empty = compare_template(empty, Image.new("L", (1, 1), 255))
        image_only = compare_template(selected, Image.new("L", (1, 1), 255))
        template_only = compare_template(empty, binary_mask(("1",)))

        self.assertIsNone(both_empty.iou)
        self.assertEqual(both_empty.union_pixels, 0)
        self.assertEqual(image_only.iou, 0.0)
        self.assertEqual(template_only.iou, 0.0)

    def test_resetting_transform_restores_centered_baseline(self):
        transform = TemplateTransform(
            2.5, offset_x=7, offset_y=-4, scale_multiplier=1.8,
            width_scale_multiplier=1.1, height_scale_multiplier=0.9,
        )
        transform.reset()

        self.assertEqual((transform.offset_x, transform.offset_y), (0, 0))
        self.assertEqual(transform.scale_multiplier, 1.0)
        self.assertEqual(transform.width_scale_multiplier, 1.0)
        self.assertEqual(transform.height_scale_multiplier, 1.0)
        self.assertEqual(transform.scale, 2.5)
        self.assertEqual(transform.scale_y, 2.5)

    def test_baseline_and_adjusted_scores_are_distinct_measurements(self):
        image_mask = binary_mask(("00000", "01000", "00000"))
        template = binary_mask(("1",))

        baseline = compare_template(image_mask, template)
        adjusted = compare_template(image_mask, template, offset_x=-1)

        self.assertEqual(baseline.iou, 0.0)
        self.assertEqual(adjusted.offset_x, -1)
        self.assertEqual(adjusted.iou, 1.0)

    def test_transfer_recenters_on_new_baseline_but_preserves_offsets_and_scales(self):
        settings = TemplateFitSettings(
            variant=TEMPLATE_VARIANT_THIN_ZERO_INDEPENDENT,
            digit="0",
            threshold=160,
            offset_x=-2,
            offset_y=3,
            scale_multiplier=1.0,
            width_scale_multiplier=0.9,
            height_scale_multiplier=1.15,
        )

        transferred = transfer_template_fit(settings, new_baseline_scale=4.0)

        self.assertEqual(transferred.baseline_scale, 4.0)
        self.assertEqual((transferred.offset_x, transferred.offset_y), (-2, 3))
        self.assertEqual(transferred.scale_multiplier, 1.0)
        self.assertEqual(transferred.width_scale_multiplier, 0.9)
        self.assertEqual(transferred.height_scale_multiplier, 1.15)
        self.assertEqual((transferred.scale, transferred.scale_y), (3.6, 4.6))
        self.assertEqual(settings.threshold, 160)

    def test_roi_report_names_half_open_coordinates_without_reordering(self):
        self.assertEqual(
            format_roi_coordinates((896, 108, 926, 152)),
            "left=896, top=108, right=926, bottom=152 (right/bottom exclusive)",
        )

    def test_masks_are_unchanged_and_repeated_comparisons_are_deterministic(self):
        image_mask = binary_mask(("101", "010"))
        template = binary_mask(("11",))
        original_image = image_mask.tobytes()
        original_template = template.tobytes()

        first = compare_template(image_mask, template, offset_x=1, scale=1.5)
        second = compare_template(image_mask, template, offset_x=1, scale=1.5)

        self.assertEqual(first.iou, second.iou)
        self.assertEqual(first.visualization.tobytes(), second.visualization.tobytes())
        self.assertEqual(image_mask.tobytes(), original_image)
        self.assertEqual(template.tobytes(), original_template)


if __name__ == "__main__":
    unittest.main()
