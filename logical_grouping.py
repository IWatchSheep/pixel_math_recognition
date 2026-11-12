"""Geometry measurements and bounded grouping hypotheses for selected components."""

from dataclasses import dataclass
from itertools import combinations
from statistics import median

from connected_sections import ConnectedSection

MAX_GROUPING_CANDIDATES = 8


@dataclass(frozen=True)
class LogicalPiece:
    identifier: str
    sections: tuple[ConnectedSection, ...]

    @property
    def bounding_rectangle(self) -> tuple[int, int, int, int]:
        if not self.sections:
            raise ValueError("A logical piece must contain at least one section.")
        left = min(section.min_x for section in self.sections)
        top = min(section.min_y for section in self.sections)
        right = max(section.max_x for section in self.sections) + 1
        bottom = max(section.max_y for section in self.sections) + 1
        return left, top, right - left, bottom - top


@dataclass(frozen=True)
class PiecePairMeasurement:
    first_piece: str
    second_piece: str
    horizontal_gap: int
    vertical_gap: int
    vertical_overlap: int
    baseline_difference: int
    first_width: int
    first_height: int
    second_width: int
    second_height: int
    width_ratio: float
    height_ratio: float
    normalized_horizontal_gap: float
    normalized_vertical_gap: float
    normalized_baseline_difference: float


@dataclass(frozen=True)
class GroupingCandidate:
    groups: tuple[tuple[str, ...], ...]
    proposed_pairs: tuple[tuple[str, str], ...]
    heuristic_score: float | None


@dataclass(frozen=True)
class GroupingAnalysis:
    pieces: tuple[LogicalPiece, ...]
    typical_piece_height: float
    pair_measurements: tuple[PiecePairMeasurement, ...]
    candidates: tuple[GroupingCandidate, ...]


@dataclass(frozen=True)
class IntendedGrouping:
    alternative_index: int
    groups: tuple[tuple[str, ...], ...]
    description: str


def combine_sections(
    identifier: str, sections: tuple[ConnectedSection, ...]
) -> LogicalPiece:
    """Make a logical piece from distinct source components without changing them."""
    if not identifier.strip():
        raise ValueError("A logical piece needs a non-empty identifier.")
    if not sections:
        raise ValueError("Select at least one connected section for a piece.")
    if len({section.number for section in sections}) != len(sections):
        raise ValueError("A connected section cannot be repeated in one piece.")
    return LogicalPiece(identifier, tuple(sorted(sections, key=lambda section: section.number)))


def measure_piece_pairs(
    pieces: tuple[LogicalPiece, ...],
) -> tuple[float, tuple[PiecePairMeasurement, ...]]:
    """Measure all unordered piece pairs; intersecting axis ranges have zero gap."""
    if not pieces:
        return 0.0, ()
    heights = [piece.bounding_rectangle[3] for piece in pieces]
    typical_height = float(median(heights))
    typical_height = max(typical_height, 1.0)
    measurements = []
    for first, second in combinations(pieces, 2):
        ax, ay, aw, ah = first.bounding_rectangle
        bx, by, bw, bh = second.bounding_rectangle
        ar, ab = ax + aw, ay + ah
        br, bb = bx + bw, by + bh
        horizontal_gap = max(0, bx - ar, ax - br)
        vertical_gap = max(0, by - ab, ay - bb)
        vertical_overlap = max(0, min(ab, bb) - max(ay, by))
        baseline_difference = abs((ab - 1) - (bb - 1))
        measurements.append(
            PiecePairMeasurement(
                first.identifier,
                second.identifier,
                horizontal_gap,
                vertical_gap,
                vertical_overlap,
                baseline_difference,
                aw,
                ah,
                bw,
                bh,
                max(aw, bw) / min(aw, bw),
                max(ah, bh) / min(ah, bh),
                horizontal_gap / typical_height,
                vertical_gap / typical_height,
                baseline_difference / typical_height,
            )
        )
    return typical_height, tuple(measurements)


def generate_grouping_candidates(
    pieces: tuple[LogicalPiece, ...],
    measurements: tuple[PiecePairMeasurement, ...],
    limit: int = MAX_GROUPING_CANDIDATES,
) -> tuple[GroupingCandidate, ...]:
    """Return all-singleton hypothesis and a bounded list of plausible pairings.

    Pair ordering uses normalized horizontal/vertical distance, baseline difference,
    and vertical overlap as descriptive geometry. It is not a correctness score.
    """
    if limit < 1:
        raise ValueError("Candidate limit must be positive.")
    identifiers = tuple(piece.identifier for piece in pieces)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Piece identifiers must be unique.")
    singleton = GroupingCandidate(tuple((identifier,) for identifier in identifiers), (), None)
    pair_scores = []
    measurement_by_pair = {
        frozenset((item.first_piece, item.second_piece)): item
        for item in measurements
    }
    for first, second in combinations(identifiers, 2):
        item = measurement_by_pair[frozenset((first, second))]
        score = (
            item.normalized_horizontal_gap
            + item.normalized_vertical_gap
            + item.normalized_baseline_difference
            - min(item.vertical_overlap / max(item.first_height, item.second_height), 1.0)
        )
        pair_scores.append((score, first, second))
    pair_scores.sort(key=lambda row: (row[0], row[1], row[2]))
    candidates = [singleton]

    def candidate_for_pairs(
        pair_set: tuple[tuple[float, str, str], ...]
    ) -> GroupingCandidate:
        paired = {identifier for _score, first, second in pair_set for identifier in (first, second)}
        groups = [(first, second) for _score, first, second in pair_set]
        groups.extend((identifier,) for identifier in identifiers if identifier not in paired)
        pairs = tuple((first, second) for _score, first, second in pair_set)
        return GroupingCandidate(tuple(groups), pairs, sum(score for score, _a, _b in pair_set))

    single_pair_slots = min(len(pair_scores), max(0, limit - 1), 4)
    candidates.extend(
        candidate_for_pairs((pair,)) for pair in pair_scores[:single_pair_slots]
    )
    double_pairs = []
    for first_pair, second_pair in combinations(pair_scores, 2):
        if {first_pair[1], first_pair[2]}.isdisjoint(
            (second_pair[1], second_pair[2])
        ):
            ordered = tuple(sorted((first_pair, second_pair), key=lambda row: (row[0], row[1], row[2])))
            double_pairs.append(ordered)
    double_pairs.sort(
        key=lambda pair_set: (
            sum(row[0] for row in pair_set),
            tuple((row[1], row[2]) for row in pair_set),
        )
    )
    candidates.extend(
        candidate_for_pairs(pair_set)
        for pair_set in double_pairs[:max(0, limit - len(candidates))]
    )
    if len(candidates) < limit:
        remaining = pair_scores[single_pair_slots:]
        candidates.extend(
            candidate_for_pairs((pair,))
            for pair in remaining[:limit - len(candidates)]
        )
    return tuple(candidates)


def analyze_grouping(
    pieces: tuple[LogicalPiece, ...],
    candidate_limit: int = MAX_GROUPING_CANDIDATES,
) -> GroupingAnalysis:
    """Compute deterministic pair measurements and bounded alternatives."""
    if len({piece.identifier for piece in pieces}) != len(pieces):
        raise ValueError("Piece identifiers must be unique.")
    typical_height, measurements = measure_piece_pairs(pieces)
    candidates = generate_grouping_candidates(pieces, measurements, candidate_limit)
    return GroupingAnalysis(pieces, typical_height, measurements, candidates)


def compare_grouping_consistency(
    measurements: tuple[PiecePairMeasurement, ...],
    paired_groups: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Describe measurement differences between user-annotated two-piece groups."""
    by_pair = {
        frozenset((item.first_piece, item.second_piece)): item
        for item in measurements
    }
    comparisons = []
    for (first_a, first_b), (second_a, second_b) in combinations(paired_groups, 2):
        first = by_pair.get(frozenset((first_a, first_b)))
        second = by_pair.get(frozenset((second_a, second_b)))
        if first is None or second is None:
            continue
        comparisons.append(
            f"{first_a}/{first_b} vs {second_a}/{second_b}: normalized horizontal-gap "
            f"difference {abs(first.normalized_horizontal_gap - second.normalized_horizontal_gap):.2f}; "
            f"baseline-difference delta "
            f"{abs(first.normalized_baseline_difference - second.normalized_baseline_difference):.2f}; "
            f"height-ratio delta {abs(first.height_ratio - second.height_ratio):.2f}."
        )
    return tuple(comparisons)


def mark_candidate_intended(
    analysis: GroupingAnalysis, candidate_index: int
) -> IntendedGrouping:
    """Create a stable, displayable annotation from the selected alternative."""
    if not 0 <= candidate_index < len(analysis.candidates):
        raise ValueError("Selected grouping alternative is no longer available.")
    groups = analysis.candidates[candidate_index].groups
    return IntendedGrouping(
        candidate_index, groups, format_grouping_description(groups)
    )


def format_grouping_description(
    groups: tuple[tuple[str, ...], ...]
) -> str:
    """Describe a grouping annotation without assigning meaning to its pieces."""
    if not groups:
        return "No pieces are included in this grouping."
    if all(len(group) == 1 for group in groups):
        identifiers = [group[0] for group in groups]
        if len(identifiers) == 1:
            return f"{identifiers[0]} remains a separate group."
        if len(identifiers) == 2:
            joined = " and ".join(identifiers)
        else:
            joined = ", ".join(identifiers[:-1]) + f", and {identifiers[-1]}"
        return f"{joined} remain separate."
    descriptions = [
        " + ".join(group) if len(group) > 1 else f"{group[0]} remains separate"
        for group in groups
    ]
    return "; ".join(descriptions) + "."
