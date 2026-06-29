"""End-to-end on a tiny network whose FORCE set is known by hand.

This is the regression anchor for stages 1-4 on the toy network.
"""

from pyoptforce import OptForce


def _driver(toy_model):
    return OptForce(
        toy_model,
        target_reaction="EX_P",
        biomass_reaction="bio",
        target_fraction=0.5,
        solver="auto",
    )


def test_pipeline_must_sets(toy_model):
    of = _driver(toy_model)
    of.compute_flux_ranges()
    ms = of.find_must_sets(max_order=2)
    assert set(ms.mustU) == {"r2", "prod", "EX_P"}
    assert set(ms.mustL) == {"r1", "bio"}


def test_force_set_toy(toy_model):
    of = _driver(toy_model)
    of.find_must_sets(max_order=2)
    fs = of.find_force_sets(k=1, n_solutions=10)

    singles = {frozenset(s["reactions"]): s for s in fs.sets}
    # up-regulating a product-branch reaction is a valid single FORCE set
    assert frozenset({"r2"}) in singles
    assert singles[frozenset({"r2"})]["type"]["r2"] == "up"
    assert singles[frozenset({"r2"})]["objective"] >= 5 - 1e-6

    # down-regulating the growth branch alone does NOT force product
    assert frozenset({"r1"}) not in singles
    assert frozenset({"bio"}) not in singles


def test_force_set_objective_meets_threshold(toy_model):
    of = _driver(toy_model)
    of.find_must_sets(max_order=2)
    fs = of.find_force_sets(k=1, n_solutions=10)
    assert len(fs) >= 1
    for s in fs.sets:
        assert s["objective"] >= of.target_threshold - 1e-6
