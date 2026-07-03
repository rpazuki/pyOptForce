"""Tests for the FVSEOF over-expression-target extension (:mod:`pyoptforce.extensions.fseof`).

Two things are pinned down here:

* the classic FVSEOF *midpoint slope* signal and its two strengthenings — our
  monotone-worst-case (``min_monotone_up``) tier and the OptForce-style *necessity*
  tier (``must_up`` = target-rung range disjoint above the WT range);
* both entry points the user cares about — *forced increase* (enforce a product/growth
  flux and rank targets) and *reverse check* (given an observed growth rate, is a named
  reaction a supported over-expression target, and how strongly).

The toy network is small enough that every verdict is hand-derivable, and the necessity
tier is cross-checked against OptForce's own MUSTU classification on the same model.
"""

from __future__ import annotations

import cobra
import pytest

from pyoptforce import fva, must_sets
from pyoptforce.model import set_target_yield
from pyoptforce.extensions.fseof import fvseof


# ============================================================================ toy net
def test_forced_product_necessity_equals_optforce_mustu(toy_model):
    """`must_up` (necessity) on an EX_P sweep must reproduce OptForce's MUSTU set.

    FVSEOF excludes the enforced reaction itself, so the equivalence is
    ``necessary_up() ∪ {enforced} == MUSTU``. This ties the cheap LP scan back to the
    rigorous stage-3 classification: both agree on *which reactions must increase*.
    """
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=6)

    # OptForce MUSTU on the same toy (WT at max growth vs EX_P >= 0.5 * max).
    tm = set_target_yield(toy_model, "EX_P", "bio", 0.5)
    fr = fva.compute_flux_ranges(toy_model, tm, fraction_of_optimum=1.0)
    mustU = set(must_sets.first_order(fr).mustU)

    assert set(res.necessary_up()) | {"EX_P"} == mustU
    # hand-known values (conftest documents MUSTU = {r2, prod, EX_P})
    assert set(res.necessary_up()) == {"r2", "prod"}


def test_min_increase_is_strictly_stronger_than_slope_weaker_than_necessity(toy_model):
    """The crux: monotonic increase != necessity.

    `up_A` (shared uptake) rises monotonically at its low end (`min_monotone_up`) and has
    a positive midpoint slope, yet is NOT necessary: the wild type can already run it at
    the top value (its WT range reaches 10), so `must_up` is False. `r2` *is* necessary.
    On this network the flags nest strictly: {must_up} ⊂ {min_monotone_up}.
    """
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=6)

    up_a = res.check("up_A")
    assert up_a.min_monotone_up is True and up_a.slope > 0  # weak + monotone signal...
    assert up_a.must_up is False                            # ...but NOT necessary
    assert up_a.verdict == "amplification"

    assert res.check("r2").must_up is True                  # r2 IS necessary

    must = {r.reaction for r in res.rows if r.must_up}
    mono = {r.reaction for r in res.rows if r.min_monotone_up}
    assert must == {"r2", "prod"}
    assert mono == {"r2", "prod", "up_A"}
    assert must < mono  # proper subset: necessity strictly stronger than min-increase


def test_forced_product_verdicts_are_hand_exact(toy_model):
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=6)
    verdicts = res.to_dataframe()["verdict"].to_dict()
    assert verdicts == {
        "r2": "must_up",
        "prod": "must_up",
        "up_A": "amplification",
        "r1": "attenuation",
        "bio": "attenuation",
    }
    # amplification and attenuation partitions are disjoint
    amp = set(res.amplification_targets().index)
    att = set(res.attenuation_targets().index)
    assert amp.isdisjoint(att)
    assert amp == {"r2", "prod", "up_A"} and att == {"r1", "bio"}


def test_enforced_reaction_excluded_from_results(toy_model):
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=5)
    assert "EX_P" not in res.to_dataframe().index
    assert "EX_P" not in res.profiles


def test_reverse_check_via_observed_growth(toy_model):
    """Enforce biomass from an 'observed' 4.0 up to max: r1 (growth branch) must rise.

    This is the reverse-check use case: given a measured growth rate, which reactions are
    forced up to reach higher growth — and is the candidate among them.
    """
    res = fvseof(toy_model, "bio", lo=4.0, n_steps=6)

    r1 = res.check("r1")
    assert r1.verdict == "must_up" and r1.must_up is True
    assert "NECESSARY" in res.interpret("r1")
    # the product branch must be given up to grow faster
    assert res.check("r2").is_attenuation
    assert res.check("prod").is_attenuation


def test_growth_default_baseline_has_no_headroom_raises(toy_model):
    """Enforcing biomass with the default baseline (= max growth) leaves nothing to scan.

    This is the capacity caveat made concrete: a plain FBA model is already at its growth
    optimum, so the observed==maximum case must fail loudly rather than silently return an
    empty scan.
    """
    with pytest.raises(ValueError, match="headroom"):
        fvseof(toy_model, "bio", biomass_reaction="bio", n_steps=6)


def test_check_unknown_reaction_raises(toy_model):
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=5)
    with pytest.raises(KeyError):
        res.check("does_not_exist")


def test_amplification_min_verdict_filter(toy_model):
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=6)
    # cutoff at must_up keeps only the necessity tier
    only_necessary = set(res.amplification_targets(min_verdict="must_up").index)
    assert only_necessary == {"r2", "prod"}
    # cutoff at amplification keeps necessity + monotone tiers (still excludes weak_up)
    with_amp = set(res.amplification_targets(min_verdict="amplification").index)
    assert with_amp == {"r2", "prod", "up_A"}
    with pytest.raises(ValueError, match="min_verdict"):
        res.amplification_targets(min_verdict="not_a_verdict")


def test_n_steps_and_reactions_validation(toy_model):
    with pytest.raises(ValueError, match="n_steps"):
        fvseof(toy_model, "EX_P", n_steps=1)
    # restricting the scan to a subset (and excluding the enforced reaction from it)
    res = fvseof(toy_model, "EX_P", reactions=["r1", "r2"], n_steps=4)
    assert set(res.to_dataframe().index) == {"r1", "r2"}


def test_profiles_align_with_levels(toy_model):
    res = fvseof(toy_model, "EX_P", biomass_reaction="bio", n_steps=7)
    assert len(res.levels) == 7
    for rid, prof in res.profiles.items():
        assert len(prof["vmin"]) == len(res.levels)
        assert len(prof["vmax"]) == len(res.levels)


# ============================================================================ e_coli
@pytest.fixture(scope="module")
def ecoli() -> cobra.Model:
    try:
        return cobra.io.load_model("e_coli_core")
    except Exception as exc:  # pragma: no cover - only without network access
        pytest.skip(f"e_coli_core unavailable (no network access?): {exc}")


_ECOLI_BIOMASS = "BIOMASS_Ecoli_core_w_GAM"


def _self_consistent(res) -> bool:
    """Every verdict is reproducible from its own flags (guards the classify logic)."""
    for r in res.rows:
        if r.must_up:
            ok = r.verdict == "must_up"
        elif r.must_down:
            ok = r.verdict == "must_down"
        elif r.min_monotone_up and r.slope > 0:
            ok = r.verdict == "amplification"
        elif r.max_monotone_down and r.slope < 0:
            ok = r.verdict == "attenuation"
        elif r.slope > 0:
            ok = r.verdict == "weak_up"
        elif r.slope < 0:
            ok = r.verdict == "weak_down"
        else:
            ok = r.verdict == "none"
        if not ok:
            return False
        # necessity flag must actually mean range disjointness
        if r.must_up and not (r.vmin_top > r.vmax_wt):
            return False
        if r.must_down and not (r.vmax_top < r.vmin_wt):
            return False
    return True


def test_ecoli_forced_succinate_scan(ecoli):
    """Forced succinate over-production names real amplification targets, consistently."""
    res = fvseof(
        ecoli, "EX_succ_e", biomass_reaction=_ECOLI_BIOMASS, n_steps=8
    )
    assert "EX_succ_e" not in res.to_dataframe().index
    assert _self_consistent(res)
    # a real network under a real product demand yields *some* directional signal
    assert len(res.amplification_targets()) + len(res.attenuation_targets()) > 0
    # necessary_up returns exactly the must_up rows
    assert set(res.necessary_up()) == {r.reaction for r in res.rows if r.must_up}
    # amplification / attenuation verdicts never collide
    assert set(res.amplification_targets().index).isdisjoint(
        set(res.attenuation_targets().index)
    )


def test_ecoli_growth_default_baseline_raises(ecoli):
    """On the real model, WT FBA growth already equals the max — no headroom (capacity)."""
    with pytest.raises(ValueError, match="headroom"):
        fvseof(ecoli, _ECOLI_BIOMASS, n_steps=6)


def test_ecoli_growth_reverse_check_with_observed_rate(ecoli):
    """Anchor at an 'observed' 30%-of-max growth and force it up: reverse-check a gene."""
    with ecoli as m:
        m.objective = _ECOLI_BIOMASS
        mu_max = float(m.slim_optimize())
    res = fvseof(ecoli, _ECOLI_BIOMASS, lo=0.3 * mu_max, n_steps=8)

    assert _self_consistent(res)
    # forcing growth 0.3x -> 1.0x must drive *some* reactions (precursor supply, uptake)
    assert len(res.amplification_targets()) + len(res.attenuation_targets()) > 0

    # reverse-check a specific reaction: returns a valid verdict + non-empty interpretation
    row = res.check("PGI")
    assert row.verdict in {
        "must_up", "must_down", "amplification", "attenuation",
        "weak_up", "weak_down", "none",
    }
    assert isinstance(res.interpret("PGI"), str) and res.interpret("PGI")
