"""Tests for stage 3. First order is pure interval logic -> easy to hand-verify."""

import cobra

from pyoptforce import fva, must_sets
from pyoptforce.model import set_target_yield
from pyoptforce.results import FluxRanges


def _toy_ranges(toy_model) -> FluxRanges:
    tm = set_target_yield(toy_model, "EX_P", "bio", 0.5)
    return fva.compute_flux_ranges(toy_model, tm, fraction_of_optimum=1.0)


def test_first_order_on_toy_network(toy_model):
    fr = _toy_ranges(toy_model)
    ms = must_sets.first_order(fr)
    # product-branch reactions must increase; growth-branch must decrease
    assert set(ms.mustU) == {"r2", "prod", "EX_P"}
    assert set(ms.mustL) == {"r1", "bio"}


def test_first_order_directions_are_exclusive(toy_model):
    fr = _toy_ranges(toy_model)
    ms = must_sets.first_order(fr)
    assert not (set(ms.mustU) & set(ms.mustL))


def test_second_order_no_spurious_pairs(toy_model):
    # Every coupled reaction is already first order here, so no pairs should appear.
    fr = _toy_ranges(toy_model)
    ms = must_sets.find_must_sets(toy_model, fr, max_order=2)
    assert ms.mustUU == [] and ms.mustLL == [] and ms.mustUL == []


def test_must_sets_dataframe(toy_model):
    fr = _toy_ranges(toy_model)
    ms = must_sets.first_order(fr)
    df = ms.to_dataframe()
    assert set(df["set"]) == {"MUSTU", "MUSTL"}


# ------------------------------------------------------------ WT joint-feasibility bug
def test_wt_feasible_with_returns_false_not_raise_on_joint_infeasibility():
    # Regression test: `slim_optimize(error_value=None)` does NOT return None on
    # failure (it raises) -- per cobra's own docstring, error_value=None means "raise
    # instead". `_wt_feasible_with` used to assume the opposite, so a genuinely
    # WT-infeasible joint bound (exactly the condition MUSTUU/LL/UL are testing for)
    # crashed `second_order()` with a raw cobra.exceptions.Infeasible instead of
    # classifying the pair. Network: a shared resource "R" (cap 5) feeds two
    # independent branches x, y (cap 10 each); x>=4 alone and y>=4 alone are each
    # feasible (uses <=4 of the 5-unit supply), but jointly x>=4 AND y>=4 needs
    # supply>=8 > 5, genuinely LP-infeasible (not merely a bound-order violation).
    m = cobra.Model("shared_bottleneck")
    r_met = cobra.Metabolite("R")
    m.add_metabolites([r_met])
    supply = cobra.Reaction("supply", lower_bound=0, upper_bound=5)
    supply.add_metabolites({r_met: 1})
    x = cobra.Reaction("x", lower_bound=0, upper_bound=10)
    x.add_metabolites({r_met: -1})
    y = cobra.Reaction("y", lower_bound=0, upper_bound=10)
    y.add_metabolites({r_met: -1})
    m.add_reactions([supply, x, y])
    m.objective = "supply"

    assert must_sets._wt_feasible_with(m, {"x": (">=", 4.0)}) is True
    assert must_sets._wt_feasible_with(m, {"y": (">=", 4.0)}) is True
    assert must_sets._wt_feasible_with(
        m, {"x": (">=", 4.0), "y": (">=", 4.0)}
    ) is False
