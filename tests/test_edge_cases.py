"""Organism-agnostic robustness: objective handling, id validation, infinite bounds.

These guard the failure modes that only show up on a model whose SBML differs from the
toy / e_coli_core happy path (different default objective, missing ids, ±inf bounds).
"""

import cobra
import pytest

from pyoptforce import OptForce
from pyoptforce import bilevel, model as M


# ----------------------------------------------------------------- objective anchoring
def test_prepare_model_feasible_despite_unbounded_default_objective():
    """A model whose default objective is unbounded is still *feasible* and must pass.

    The old optimise()-based check raised 'not optimisable' here; the constant-objective
    feasibility probe does not.
    """
    m = cobra.Model("unbounded_obj")
    a = cobra.Metabolite("A")
    m.add_metabolites([a])
    r_in = cobra.Reaction("in", lower_bound=0, upper_bound=float("inf"))
    r_in.add_metabolites({a: 1})
    r_out = cobra.Reaction("out", lower_bound=0, upper_bound=float("inf"))
    r_out.add_metabolites({a: -1})
    m.add_reactions([r_in, r_out])
    m.objective = "in"  # unbounded maximise

    assert M.is_feasible(m)
    prepared = M.prepare_model(m)            # must not raise
    assert len(prepared.reactions) == 2


def test_objective_anchored_regardless_of_sbml_default(toy_model):
    # sabotage the default objective; OptForce must still anchor WT to biomass
    toy_model.objective = "up_A"
    of = OptForce(toy_model, target_reaction="EX_P", biomass_reaction="bio",
                  target_fraction=0.5)
    assert "bio" in str(of.model.objective.expression)
    of.compute_flux_ranges()
    # WT basal state pins growth branch (bio at max), proving the anchor took effect
    assert abs(of.flux_ranges.max_w["r1"] - 10) < 1e-6
    assert abs(of.flux_ranges.max_w["r2"]) < 1e-6


# ------------------------------------------------------------------------- id validation
def test_missing_target_id_raises(toy_model):
    with pytest.raises(KeyError):
        OptForce(toy_model, target_reaction="does_not_exist", biomass_reaction="bio")


def test_missing_biomass_id_raises(toy_model):
    with pytest.raises(KeyError):
        OptForce(toy_model, target_reaction="EX_P", biomass_reaction="does_not_exist")


def test_target_equals_biomass_raises(toy_model):
    with pytest.raises(ValueError):
        OptForce(toy_model, target_reaction="bio", biomass_reaction="bio")


def test_fraction_out_of_range_raises(toy_model):
    with pytest.raises(ValueError):
        OptForce(toy_model, target_reaction="EX_P", biomass_reaction="bio",
                 wt_growth_fraction=1.5)


# --------------------------------------------------------------- producibility / growth
def test_non_producible_target_raises(toy_model):
    toy_model.reactions.EX_P.upper_bound = 0.0  # product cannot leave
    of = OptForce(toy_model, target_reaction="EX_P", biomass_reaction="bio",
                  target_fraction=0.5)
    with pytest.raises(ValueError):
        of.compute_flux_ranges()


def test_non_growing_model_raises(toy_model):
    toy_model.reactions.up_A.upper_bound = 0.0  # no carbon in -> no growth, no product
    of = OptForce(toy_model, target_reaction="EX_P", biomass_reaction="bio",
                  target_fraction=0.5)
    with pytest.raises(ValueError):
        of.compute_flux_ranges()


# ------------------------------------------------------------------ infinite-bound MILP
def test_finite_or_raise_surrogate_and_error():
    fb = {"r": (-3.0, 7.0)}
    assert bilevel._finite_or_raise(fb, "r", 0, "lower") == -3.0
    assert bilevel._finite_or_raise(fb, "r", 1, "upper") == 7.0
    with pytest.raises(ValueError):
        bilevel._finite_or_raise(None, "r", 0, "lower")
    with pytest.raises(ValueError):
        bilevel._finite_or_raise({"r": (float("-inf"), 7.0)}, "r", 0, "lower")


# ------------------------------------------------------------ slim_optimize(error_value=None) misuse
# Regression tests: `slim_optimize(error_value=None)` does NOT return None on failure —
# per cobra's own docstring, error_value=None means "raise instead". `is_feasible`,
# `theoretical_max`, `_worst_case_target`, and `_wt_feasible_with` (must_sets.py) all
# used to assume the opposite, so a genuinely infeasible/unbounded LP would crash with a
# raw cobra exception instead of the clean, intended failure signal. Fixed by using the
# default NaN sentinel and checking `math.isnan`. See must_sets.py's own regression test
# (test_must_sets.py) for the `_wt_feasible_with` half of this bug.
def test_is_feasible_returns_false_not_raise_on_infeasible_model():
    # genuinely LP-infeasible (mass balance forces r_in == r_out, but their ranges
    # [5,10] and [0,2] cannot overlap) -- not merely an unbounded default objective.
    m = cobra.Model("infeasible")
    a = cobra.Metabolite("A")
    m.add_metabolites([a])
    r_in = cobra.Reaction("in", lower_bound=5, upper_bound=10)
    r_in.add_metabolites({a: 1})
    r_out = cobra.Reaction("out", lower_bound=0, upper_bound=2)
    r_out.add_metabolites({a: -1})
    m.add_reactions([r_in, r_out])
    m.objective = "in"

    assert M.is_feasible(m) is False


def test_theoretical_max_raises_clean_valueerror_not_raw_cobra_exception():
    m = cobra.Model("unbounded")
    a = cobra.Metabolite("A")
    m.add_metabolites([a])
    r_in = cobra.Reaction("in", lower_bound=0, upper_bound=float("inf"))
    r_in.add_metabolites({a: 1})
    r_out = cobra.Reaction("out", lower_bound=0, upper_bound=float("inf"))
    r_out.add_metabolites({a: -1})
    m.add_reactions([r_in, r_out])

    with pytest.raises(ValueError, match="Could not optimise"):
        M.theoretical_max(m, "in")


def test_worst_case_target_handles_biomass_down_conflict_gracefully(toy_model):
    # Regression test: at a high enough target_fraction, the biomass reaction's own
    # M-derived max ("down" intervention value) falls below the viability floor already
    # applied to its lower bound. Setting both on the SAME reaction used to raise a raw
    # cobra ValueError (lb > ub) instead of being treated as "no engineered strain
    # exists for this combination" (None).
    of = OptForce(toy_model, target_reaction="EX_P", biomass_reaction="bio",
                  target_fraction=0.99, solver="auto")
    of.compute_flux_ranges()
    assert of.flux_ranges.max_m["bio"] < of.min_biomass_fraction * of.max_growth
    assert of._worst_case_target({"bio": "down"}) is None
