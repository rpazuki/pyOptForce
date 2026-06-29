"""Tests for stage 3. First order is pure interval logic -> easy to hand-verify."""

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
