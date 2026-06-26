"""Stage 3: MUST-set classification (first and higher order).

First order (MUSTU/MUSTL) is pure interval comparison between WT and target ranges.
Higher orders (MUSTUU/LL/UL, ...) require the bilevel MILP in bilevel.py.
"""

from __future__ import annotations

from pyoptforce.results import FluxRanges, MustSets


def first_order(ranges: FluxRanges) -> MustSets:
    """MUSTU / MUSTL via interval logic.

    MUSTU: max_m < min_w  -> ... (decide exact rule and document vs paper Eq.)
    MUSTL: min_m > max_w  -> ...
    NB: confirm the inequality directions against Ranganathan et al. 2010.
    """
    raise NotImplementedError


def second_order(ranges: FluxRanges, must: MustSets, *, solver: str) -> MustSets:
    """MUSTUU / MUSTLL / MUSTUL via bilevel MILP (uses bilevel.solve_bilevel).

    Excludes reactions already in first-order sets. Enumerate via integer cuts.
    """
    raise NotImplementedError


def find_must_sets(ranges: FluxRanges, *, max_order: int = 2, solver: str) -> MustSets:
    """Driver: first order, then higher orders up to max_order."""
    raise NotImplementedError
