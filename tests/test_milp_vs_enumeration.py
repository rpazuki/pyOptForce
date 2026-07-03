"""Thorough cross-validation: does the MILP FORCE-set path agree with the
enumerate-and-verify LP path?

Scope: this compares ``OptForce.find_force_sets(method="milp")`` against
``method="enumerate"`` (stage 4 only) -- the only place in this codebase with two
independent solution paths for the same optimisation problem. (Stage 3's second-order
MUST classification has a single LP-feasibility implementation, ``must_sets.py``; there
is no MILP alternative to cross-check it against. See ``bilevel.py``'s own isolated
duality self-test in ``test_bilevel.py`` for that keystone's unit coverage, and
``test_gurobi.py`` for the original minimal smoke test this file supersedes in depth.)

Methodology
-----------
The two methods promise *different* things by design, not just different algorithms:

* ``enumerate`` returns INCLUSION-MINIMAL valid sets, smallest-size first, and stops
  after ``n_solutions`` distinct minimal sets are found.
* ``milp`` maximises the worst-case objective for <=k interventions and enumerates
  alternate optima (via integer cuts) in decreasing-objective order. Ties at the same
  objective commonly include supersets of an already-valid smaller set (turning on an
  additional intervention that happens not to hurt the worst case is still an
  *optimal* solution to the MILP, even though it is not *minimal*).

So a naive ``set(milp.sets) == set(enum.sets)`` is only guaranteed to hold at k=1
(where "minimal" is automatic -- a singleton has no non-empty proper subset). This is
verified empirically below (see the k=2/k=3 alternate-optima counts on both the toy
network and e_coli_core). For k>=2 the correct cross-check has four independent legs:

  1. every solution either method reports has its claimed objective *independently
     re-verified* by directly re-solving the same worst-case LP (``_worst_case_target``)
     -- so a bug in either path's own bookkeeping cannot go unnoticed;
  2. filtering the MILP output down to its own inclusion-minimal members must equal the
     ``enumerate`` output exactly;
  3. the MILP's best (first-reported) objective at cardinality k must equal the true
     optimum found by an independent brute-force scan of *every* subset of size <=k (not
     just the ones either method happens to report) -- this is the ground-truth check on
     the MILP formulation itself, bypassing both methods' own bookkeeping;
  4. the ``enumerate`` output must equal the inclusion-minimal subsets of that same
     brute-force scan's valid sets.

Objective comparisons use a loose-ish tolerance (Gurobi's default MIPGap is 1e-4
relative) rather than the tight 1e-6 used for exact-LP-vs-itself checks.
"""

from __future__ import annotations

import itertools
import math

import cobra
import pytest

from pyoptforce import OptForce
from pyoptforce import solvers

pytestmark = pytest.mark.skipif(
    not solvers.gurobi_available(), reason="gurobipy not installed"
)

_TOL = 1e-6
_MILP_TOL = 1e-3  # generous vs. Gurobi's default 1e-4 relative MIPGap


# --------------------------------------------------------------------------- helpers
def _brute_force_scan(of: OptForce, directions: dict[str, str], k: int) -> dict:
    """Ground truth: worst-case objective of EVERY candidate subset of size <=k.

    Bypasses both `find_force_sets` code paths entirely -- it only calls the shared
    inner-LP primitive (`_worst_case_target`) that `enumerate` itself is built from, in
    a plain triple-nested loop with no early-exit or minimality bookkeeping. Returns
    ``{frozenset(reactions): objective_or_None}``.
    """
    results: dict[frozenset[str], float | None] = {}
    names = list(directions)
    for size in range(1, k + 1):
        for combo in itertools.combinations(names, size):
            interventions = {r: directions[r] for r in combo}
            results[frozenset(combo)] = of._worst_case_target(interventions)
    return results


def _valid(scan: dict, threshold: float, tol: float = _TOL) -> dict:
    return {s: v for s, v in scan.items() if v is not None and v >= threshold - tol}


def _minimal(valid_sets) -> set[frozenset[str]]:
    """Inclusion-minimal members: no other member is a proper subset."""
    keys = list(valid_sets)
    return {s for s in keys if not any(t < s for t in keys)}


def _as_sets(force_sets) -> set[frozenset[str]]:
    return {frozenset(s["reactions"]) for s in force_sets.sets}


# ============================================================================== toy
def _toy_driver(toy_model, target_fraction: float = 0.5) -> OptForce:
    of = OptForce(
        toy_model,
        target_reaction="EX_P",
        biomass_reaction="bio",
        target_fraction=target_fraction,
        solver="gurobi",
    )
    of.find_must_sets(max_order=2)
    return of


@pytest.mark.parametrize("k", [1, 2, 3])
def test_toy_milp_optimum_matches_brute_force(toy_model, k):
    of = _toy_driver(toy_model)
    directions = of._candidate_directions()
    valid = _valid(_brute_force_scan(of, directions, k), of.target_threshold)
    best_brute = max(valid.values()) if valid else None

    milp = of.find_force_sets(k=k, n_solutions=50, method="milp")
    best_milp = milp.sets[0]["objective"] if milp.sets else None

    if best_brute is None:
        assert best_milp is None
    else:
        assert best_milp is not None
        assert math.isclose(best_milp, best_brute, rel_tol=_MILP_TOL, abs_tol=_MILP_TOL)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_toy_enumerate_matches_brute_force_minimal_sets(toy_model, k):
    of = _toy_driver(toy_model)
    directions = of._candidate_directions()
    valid = _valid(_brute_force_scan(of, directions, k), of.target_threshold)

    enum = of.find_force_sets(k=k, n_solutions=50, method="enumerate")
    assert _as_sets(enum) == _minimal(valid)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_toy_milp_minimal_filtered_matches_enumerate(toy_model, k):
    of = _toy_driver(toy_model)
    enum = of.find_force_sets(k=k, n_solutions=50, method="enumerate")
    milp = of.find_force_sets(k=k, n_solutions=50, method="milp")

    milp_sets = [frozenset(s["reactions"]) for s in milp.sets]
    milp_minimal = {s for s in milp_sets if not any(t < s for t in milp_sets)}
    assert milp_minimal == _as_sets(enum)


def test_toy_k1_exact_equality_including_ties(toy_model):
    # at k=1 "minimal" is automatic (a singleton has no non-empty proper subset), so
    # the raw outputs -- not just their minimal-filtered forms -- must match exactly,
    # including all three tied alternatives (r2/prod/EX_P are the same physical flux).
    of = _toy_driver(toy_model)
    enum = of.find_force_sets(k=1, n_solutions=50, method="enumerate")
    milp = of.find_force_sets(k=1, n_solutions=50, method="milp")
    expected = {frozenset({"r2"}), frozenset({"prod"}), frozenset({"EX_P"})}
    assert _as_sets(enum) == expected
    assert _as_sets(milp) == expected


def test_toy_alternate_optima_diverge_at_k2_as_expected(toy_model):
    # documents *why* raw list equality is not the right check for k>=2: MILP's
    # integer-cut enumeration returns many non-minimal supersets tied at the same
    # objective, which `enumerate` correctly omits by design.
    of = _toy_driver(toy_model)
    enum = of.find_force_sets(k=2, n_solutions=50, method="enumerate")
    milp = of.find_force_sets(k=2, n_solutions=50, method="milp")
    assert len(enum.sets) == 3
    assert len(milp.sets) > len(enum.sets)
    assert all(any(m >= e for e in _as_sets(enum)) for m in _as_sets(milp))


def test_toy_every_milp_solution_independently_reverified(toy_model):
    of = _toy_driver(toy_model)
    milp = of.find_force_sets(k=3, n_solutions=50, method="milp")
    directions = of._candidate_directions()
    assert len(milp.sets) > 0
    for s in milp.sets:
        interventions = {r: directions[r] for r in s["reactions"]}
        recomputed = of._worst_case_target(interventions)
        assert recomputed is not None
        assert math.isclose(recomputed, s["objective"], rel_tol=_MILP_TOL, abs_tol=_MILP_TOL)
        assert recomputed >= of.target_threshold - _TOL


def test_toy_every_enumerate_solution_independently_reverified(toy_model):
    of = _toy_driver(toy_model)
    enum = of.find_force_sets(k=3, n_solutions=50, method="enumerate")
    directions = of._candidate_directions()
    assert len(enum.sets) > 0
    for s in enum.sets:
        interventions = {r: directions[r] for r in s["reactions"]}
        recomputed = of._worst_case_target(interventions)
        assert recomputed is not None
        assert math.isclose(recomputed, s["objective"], abs_tol=_TOL)
        assert recomputed >= of.target_threshold - _TOL


def test_toy_k0_no_interventions_both_agree_empty(toy_model):
    of = _toy_driver(toy_model)
    enum = of.find_force_sets(k=0, n_solutions=10, method="enumerate")
    milp = of.find_force_sets(k=0, n_solutions=10, method="milp")
    assert enum.sets == []
    assert milp.sets == []


@pytest.mark.parametrize("k", [1, 2, 3])
def test_toy_infeasible_engineered_strain_both_agree_empty(toy_model, k):
    # Regression scenario for the two bugs fixed in optforce.py/must_sets.py: at
    # target_fraction=0.99 every "up" candidate's forced value conflicts with the
    # viability floor, so NO valid FORCE set exists at any k. Both methods used to
    # crash here (slim_optimize misuse; bound-order conflict on the biomass reaction);
    # now they must gracefully agree on "no solution".
    of = _toy_driver(toy_model, target_fraction=0.99)
    enum = of.find_force_sets(k=k, n_solutions=10, method="enumerate")
    milp = of.find_force_sets(k=k, n_solutions=10, method="milp")
    assert enum.sets == []
    assert milp.sets == []


def test_toy_negative_controls_never_appear(toy_model):
    # down-regulating the growth branch alone (or in the {r1,bio} pair) does not force
    # product -- both methods must agree these are never part of a valid solution.
    of = _toy_driver(toy_model)
    for k in (1, 2, 3):
        milp = of.find_force_sets(k=k, n_solutions=50, method="milp")
        enum = of.find_force_sets(k=k, n_solutions=50, method="enumerate")
        for sets in (_as_sets(milp), _as_sets(enum)):
            assert frozenset({"r1"}) not in sets
            assert frozenset({"bio"}) not in sets
            assert frozenset({"r1", "bio"}) not in sets


# =========================================================================== e_coli
@pytest.fixture(scope="module")
def ecoli_succinate_driver() -> OptForce:
    """Real genome-scale-ish model, computed once and reused across this module.

    max_order=1 keeps MUST-set discovery fast (pure interval logic on top of the
    already-computed FVA ranges); second-order pairwise LPs are not needed for a stage-4
    (FORCE-set) comparison. Restricting to first order also matches
    examples/ecoli_succinate.py's own tradeoff.
    """
    try:
        model = cobra.io.load_model("e_coli_core")
    except Exception as exc:  # pragma: no cover - only hit without network access
        pytest.skip(f"e_coli_core unavailable (no network access?): {exc}")
    of = OptForce(
        model,
        target_reaction="EX_succ_e",
        biomass_reaction="BIOMASS_Ecoli_core_w_GAM",
        target_fraction=0.3,
        solver="gurobi",
    )
    of.compute_flux_ranges()
    of.find_must_sets(max_order=1)
    return of


def test_ecoli_candidate_pool_sanity(ecoli_succinate_driver):
    # sanity-check the fixture: a small, mixed up/down candidate pool including the
    # biomass reaction itself as a "down" candidate (exercises the bound-conflict fix).
    directions = ecoli_succinate_driver._candidate_directions()
    assert 3 <= len(directions) <= 10
    assert "up" in directions.values()
    assert "down" in directions.values()
    assert directions.get("BIOMASS_Ecoli_core_w_GAM") == "down"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_ecoli_milp_optimum_matches_brute_force(ecoli_succinate_driver, k):
    of = ecoli_succinate_driver
    directions = of._candidate_directions()
    valid = _valid(_brute_force_scan(of, directions, k), of.target_threshold)
    best_brute = max(valid.values()) if valid else None

    milp = of.find_force_sets(k=k, n_solutions=50, method="milp")
    best_milp = milp.sets[0]["objective"] if milp.sets else None

    if best_brute is None:
        assert best_milp is None
    else:
        assert best_milp is not None
        assert math.isclose(best_milp, best_brute, rel_tol=_MILP_TOL, abs_tol=_MILP_TOL)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_ecoli_enumerate_matches_brute_force_minimal_sets(ecoli_succinate_driver, k):
    of = ecoli_succinate_driver
    directions = of._candidate_directions()
    valid = _valid(_brute_force_scan(of, directions, k), of.target_threshold)

    enum = of.find_force_sets(k=k, n_solutions=50, method="enumerate")
    assert _as_sets(enum) == _minimal(valid)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_ecoli_milp_minimal_filtered_matches_enumerate(ecoli_succinate_driver, k):
    of = ecoli_succinate_driver
    enum = of.find_force_sets(k=k, n_solutions=50, method="enumerate")
    milp = of.find_force_sets(k=k, n_solutions=50, method="milp")

    milp_sets = [frozenset(s["reactions"]) for s in milp.sets]
    milp_minimal = {s for s in milp_sets if not any(t < s for t in milp_sets)}
    assert milp_minimal == _as_sets(enum)


def test_ecoli_k1_exact_equality(ecoli_succinate_driver):
    of = ecoli_succinate_driver
    enum = of.find_force_sets(k=1, n_solutions=50, method="enumerate")
    milp = of.find_force_sets(k=1, n_solutions=50, method="milp")
    assert _as_sets(enum) == _as_sets(milp)
    assert len(enum.sets) >= 1


def test_ecoli_every_milp_solution_independently_reverified(ecoli_succinate_driver):
    of = ecoli_succinate_driver
    directions = of._candidate_directions()
    milp = of.find_force_sets(k=3, n_solutions=50, method="milp")
    assert len(milp.sets) > 0
    for s in milp.sets:
        interventions = {r: directions[r] for r in s["reactions"]}
        recomputed = of._worst_case_target(interventions)
        assert recomputed is not None
        assert math.isclose(recomputed, s["objective"], rel_tol=_MILP_TOL, abs_tol=_MILP_TOL)
        assert recomputed >= of.target_threshold - _TOL


def test_ecoli_every_enumerate_solution_independently_reverified(ecoli_succinate_driver):
    of = ecoli_succinate_driver
    directions = of._candidate_directions()
    enum = of.find_force_sets(k=3, n_solutions=50, method="enumerate")
    assert len(enum.sets) > 0
    for s in enum.sets:
        interventions = {r: directions[r] for r in s["reactions"]}
        recomputed = of._worst_case_target(interventions)
        assert recomputed is not None
        assert math.isclose(recomputed, s["objective"], abs_tol=_TOL)
        assert recomputed >= of.target_threshold - _TOL


def test_ecoli_biomass_down_candidate_handled_gracefully(ecoli_succinate_driver):
    # regression: BIOMASS_Ecoli_core_w_GAM is itself a MUSTL/"down" candidate here;
    # applying the viability floor then the down-bound on the SAME reaction used to
    # raise a raw cobra ValueError instead of a clean feasibility result.
    of = ecoli_succinate_driver
    result = of._worst_case_target({"BIOMASS_Ecoli_core_w_GAM": "down"})
    assert result is None or isinstance(result, float)
