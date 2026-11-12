import unittest
from dataclasses import FrozenInstanceError

from PIL import Image

from connected_sections import ConnectedSection
from logical_grouping import (
    LogicalPiece,
    analyze_grouping,
    combine_sections,
    compare_grouping_consistency,
    format_grouping_description,
    mark_candidate_intended,
    measure_piece_pairs,
)
from logical_grouping_instructions import (
    GROUPING_KEYS,
    parse_logical_grouping_config,
    run_logical_grouping_experiment,
)


def section(number, left, top, width, height):
    return ConnectedSection(
        number, width * height, left, left + width - 1, top, top + height - 1
    )


CONFIG = """\
experiment=logical_grouping
roi_left=0
roi_top=0
roi_right=12
roi_bottom=5
brightness_threshold=20
pieces=piece_a=1,1|3,1;piece_b=6,1;piece_c=9,1;piece_d=11,1
intended_groups=group_left=piece_a,piece_b;group_middle=piece_c;group_right=piece_d
"""


class LogicalGroupingTests(unittest.TestCase):
    def test_combine_sections_and_piece_bounding_box(self):
        first = section(1, 2, 3, 2, 2)
        second = section(2, 8, 1, 1, 4)

        piece = combine_sections("piece_x", (first, second))

        self.assertEqual(piece.sections, (first, second))
        self.assertEqual(piece.bounding_rectangle, (2, 1, 7, 4))
        self.assertEqual(first.bounding_rectangle, (2, 3, 2, 2))
        self.assertEqual(second.bounding_rectangle, (8, 1, 1, 4))

    def test_piece_rejects_empty_or_duplicate_components(self):
        with self.assertRaises(ValueError):
            combine_sections("empty", ())
        component = section(1, 0, 0, 1, 1)
        with self.assertRaises(ValueError):
            combine_sections("duplicate", (component, component))

    def test_pair_spacing_alignment_size_and_normalization(self):
        first = combine_sections("a", (section(1, 0, 2, 2, 4),))
        second = combine_sections("b", (section(2, 5, 3, 3, 2),))

        typical, measurements = measure_piece_pairs((first, second))
        measured = measurements[0]

        self.assertEqual(typical, 3.0)
        self.assertEqual(measured.horizontal_gap, 3)
        self.assertEqual(measured.vertical_gap, 0)
        self.assertEqual(measured.vertical_overlap, 2)
        self.assertEqual(measured.baseline_difference, 1)
        self.assertEqual((measured.first_width, measured.first_height), (2, 4))
        self.assertEqual((measured.second_width, measured.second_height), (3, 2))
        self.assertEqual(measured.width_ratio, 1.5)
        self.assertEqual(measured.height_ratio, 2.0)
        self.assertEqual(measured.normalized_horizontal_gap, 1.0)

    def test_overlapping_boxes_never_report_negative_gaps(self):
        first = combine_sections("a", (section(1, 0, 0, 4, 4),))
        second = combine_sections("b", (section(2, 2, 2, 4, 4),))
        _typical, measurements = measure_piece_pairs((first, second))

        self.assertGreaterEqual(measurements[0].horizontal_gap, 0)
        self.assertEqual(measurements[0].vertical_gap, 0)
        self.assertEqual(measurements[0].vertical_overlap, 2)

    def test_two_intended_pairs_can_be_compared_without_classification(self):
        pieces = (
            combine_sections("a", (section(1, 0, 0, 2, 4),)),
            combine_sections("b", (section(2, 4, 0, 2, 4),)),
            combine_sections("c", (section(3, 10, 1, 2, 4),)),
            combine_sections("d", (section(4, 14, 1, 2, 4),)),
        )
        _typical, measurements = measure_piece_pairs(pieces)

        comparison = compare_grouping_consistency(
            measurements, (("a", "b"), ("c", "d"))
        )

        self.assertEqual(len(comparison), 1)
        self.assertIn("normalized horizontal-gap difference 0.00", comparison[0])

    def test_single_piece_group_candidate_is_retained(self):
        piece = combine_sections("only", (section(1, 0, 0, 2, 3),))
        result = analyze_grouping((piece,))

        self.assertEqual(result.pair_measurements, ())
        self.assertEqual(result.candidates[0].groups, (("only",),))
        self.assertEqual(result.candidates[0].proposed_pairs, ())

    def test_candidate_alternatives_are_bounded_and_deterministic(self):
        pieces = tuple(
            combine_sections(f"p{index}", (section(index, index * 3, 0, 2, 4),))
            for index in range(1, 7)
        )
        first = analyze_grouping(pieces, candidate_limit=4)
        second = analyze_grouping(pieces, candidate_limit=4)

        self.assertEqual(first, second)
        self.assertEqual(len(first.candidates), 4)
        self.assertEqual(
            first.candidates[0].groups,
            tuple((f"p{i}",) for i in range(1, 7)),
        )
        for candidate in first.candidates[1:]:
            self.assertTrue(any(len(group) == 2 for group in candidate.groups))
            self.assertTrue(all(len(group) <= 2 for group in candidate.groups))
            self.assertEqual(
                sorted(piece for group in candidate.groups for piece in group),
                [f"p{i}" for i in range(1, 7)],
            )

    def test_mark_all_separate_candidate_records_and_formats_intended_grouping(self):
        pieces = tuple(
            combine_sections(name, (section(index, index * 4, 0, 2, 4),))
            for index, name in enumerate(("piece_1", "piece_2", "piece_3"), start=1)
        )
        analysis = analyze_grouping(pieces)

        intended = mark_candidate_intended(analysis, 0)

        self.assertEqual(intended.alternative_index, 0)
        self.assertEqual(
            intended.groups,
            (("piece_1",), ("piece_2",), ("piece_3",)),
        )
        self.assertEqual(
            intended.description,
            "piece_1, piece_2, and piece_3 remain separate.",
        )
        self.assertEqual(mark_candidate_intended(analysis, 0), intended)
        self.assertEqual(
            format_grouping_description(intended.groups),
            intended.description,
        )

    def test_mark_candidate_intended_rejects_stale_alternative_index(self):
        analysis = analyze_grouping(
            (combine_sections("piece_1", (section(1, 0, 0, 1, 1),)),)
        )

        with self.assertRaisesRegex(ValueError, "no longer available"):
            mark_candidate_intended(analysis, 1)

    def test_parser_accepts_coordinate_seeded_pieces_and_intended_annotations(self):
        config = parse_logical_grouping_config(CONFIG, (12, 5))

        self.assertEqual(config.experiment, "logical_grouping")
        self.assertEqual(config.roi, (0, 0, 12, 5))
        self.assertEqual(config.pieces[0], ("piece_a", ((1, 1), (3, 1))))
        self.assertEqual(config.intended_groups[1], ("group_middle", ("piece_c",)))
        self.assertEqual(
            {line.split("=", 1)[0] for line in CONFIG.splitlines()},
            GROUPING_KEYS,
        )

    def test_parser_rejects_missing_unknown_duplicate_and_malformed_values(self):
        invalid = (
            CONFIG.replace("pieces=", ""),
            CONFIG + "unexpected=yes\n",
            CONFIG + "brightness_threshold=20\n",
            CONFIG.replace("piece_a=1,1|3,1", "piece_a=bad"),
            CONFIG.replace("piece_a=1,1|3,1", "piece_a=1,1;piece_a=3,1"),
            CONFIG.replace("group_middle=piece_c", "group_middle=unknown_piece"),
            CONFIG.replace("group_middle=piece_c;", ""),
        )
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_logical_grouping_config(text, (12, 5))

    def test_parser_rejects_out_of_roi_seed_or_invalid_roi(self):
        with self.assertRaisesRegex(ValueError, "inside the ROI"):
            parse_logical_grouping_config(CONFIG.replace("1,1|3,1", "12,1|3,1"), (12, 5))
        with self.assertRaisesRegex(ValueError, "positive width"):
            parse_logical_grouping_config(CONFIG.replace("roi_right=12", "roi_right=0"), (12, 5))

    def test_execution_resolves_components_and_preserves_original_image(self):
        rgb = Image.new("RGB", (12, 5), (25, 30, 35))
        grayscale = Image.new("L", (12, 5), 255)
        for point in ((1, 1), (3, 1), (6, 1), (9, 1), (11, 1)):
            grayscale.putpixel(point, 0)
        rgb_before = rgb.tobytes()
        gray_before = grayscale.tobytes()
        config = parse_logical_grouping_config(CONFIG, rgb.size)

        result = run_logical_grouping_experiment(config, rgb, grayscale)

        self.assertEqual(result.analysis.pieces[0].bounding_rectangle, (1, 1, 3, 1))
        self.assertEqual(result.analysis.typical_piece_height, 1)
        self.assertEqual(len(result.analysis.candidates), 8)
        self.assertEqual(rgb.tobytes(), rgb_before)
        self.assertEqual(grayscale.tobytes(), gray_before)

    def test_execution_rejects_seeds_not_selected_at_threshold(self):
        rgb = Image.new("RGB", (12, 5), (255, 255, 255))
        grayscale = Image.new("L", (12, 5), 255)
        config = parse_logical_grouping_config(CONFIG, rgb.size)
        with self.assertRaisesRegex(ValueError, "not selected"):
            run_logical_grouping_experiment(config, rgb, grayscale)

    def test_manual_piece_dataclass_is_immutable(self):
        piece = LogicalPiece("p", (section(1, 0, 0, 1, 1),))
        with self.assertRaises(FrozenInstanceError):
            setattr(piece, "identifier", "changed")


if __name__ == "__main__":
    unittest.main()
