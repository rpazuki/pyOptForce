"""Gurobi MILP FORCE path. Skipped unless gurobipy is installed.

When Gurobi is present the single-level strong-duality MILP must reproduce the exact
enumerate-and-verify result on the hand-checked toy network.
"""

import pytest

from pyoptforce import OptForce
from pyoptforce import solvers

pytestmark = pytest.mark.skipif(
    not solvers.gurobi_available(), reason="gurobipy not installed"
)


def _driver(toy_model):
    return OptForce(
        toy_model,
        target_reaction="EX_P",
        biomass_reaction="bio",
        target_fraction=0.5,
        solver="gurobi",
    )


def test_milp_force_matches_enumeration(toy_model):
    of = _driver(toy_model)
    of.find_must_sets(max_order=2)

    milp = of.find_force_sets(k=1, n_solutions=10, method="milp")
    enum = of.find_force_sets(k=1, n_solutions=10, method="enumerate")

    milp_sets = {frozenset(s["reactions"]) for s in milp.sets}
    enum_sets = {frozenset(s["reactions"]) for s in enum.sets}
    assert milp_sets == enum_sets

    # the up-regulation single intervention is valid; growth-branch alone is not
    assert frozenset({"r2"}) in milp_sets
    assert frozenset({"r1"}) not in milp_sets
    for s in milp.sets:
        assert s["objective"] >= of.target_threshold - 1e-6


def test_milp_requires_gurobi_message():
    # require_gurobi returns the module when present (smoke).
    assert solvers.require_gurobi() is not None
