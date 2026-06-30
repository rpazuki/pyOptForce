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
