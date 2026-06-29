"""Tests for stages 1 & 2. Compare against cobra's own FVA."""

from cobra.flux_analysis import flux_variability_analysis

from pyoptforce import fva
from pyoptforce.model import set_target_yield


def test_wild_type_ranges_matches_cobra(toy_model):
    # WT at max growth (fraction_of_optimum=1.0) must match cobra FVA exactly.
    ours = fva.wild_type_ranges(toy_model, fraction_of_optimum=1.0)
    ref = flux_variability_analysis(toy_model, fraction_of_optimum=1.0)
    for rid, (lo, hi) in ours.items():
        assert abs(lo - ref.loc[rid, "minimum"]) < 1e-6
        assert abs(hi - ref.loc[rid, "maximum"]) < 1e-6


def test_wt_basal_state_pins_growth_branch(toy_model):
    r = fva.wild_type_ranges(toy_model, fraction_of_optimum=1.0)
    # at max growth all carbon goes to B: r1=10, r2/prod/EX_P = 0
    assert abs(r["r1"][0] - 10) < 1e-6 and abs(r["r1"][1] - 10) < 1e-6
    assert abs(r["r2"][1]) < 1e-6
    assert abs(r["EX_P"][1]) < 1e-6


def test_target_ranges_force_product(toy_model):
    tm = set_target_yield(toy_model, "EX_P", "bio", 0.5)
    r = fva.target_ranges(tm)
    # EX_P >= 5 forces r2 >= 5 and r1 <= 5
    assert r["EX_P"][0] >= 5 - 1e-6
    assert r["r2"][0] >= 5 - 1e-6
    assert r["r1"][1] <= 5 + 1e-6


def test_compute_flux_ranges_packs_both(toy_model):
    tm = set_target_yield(toy_model, "EX_P", "bio", 0.5)
    fr = fva.compute_flux_ranges(toy_model, tm, fraction_of_optimum=1.0)
    df = fr.to_dataframe()
    assert {"min_w", "max_w", "min_m", "max_m"} <= set(df.columns)
    assert "r2" in df.index
