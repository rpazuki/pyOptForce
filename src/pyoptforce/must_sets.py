"""Stage 3: MUST-set classification (first and higher order).

First order (MUSTU/MUSTL) is pure interval comparison between WT and target ranges.
Higher orders (MUSTUU/LL/UL) detect *joint* coupling that single-reaction FVA misses.

Inequality directions (confirmed against Ranganathan et al. 2010, §MUST sets)
----------------------------------------------------------------------------
With WT range ``[min_w, max_w]`` and target range ``[min_m, max_m]`` for reaction j:

* ``MUSTU`` (flux must INCREASE):  ``min_m > max_w``
  — even the lowest target flux exceeds the highest WT flux, so j is forced up.
* ``MUSTL`` (flux must DECREASE):  ``max_m < min_w``
  — even the highest target flux is below the lowest WT flux, so j is forced down.

Second order (re-derivation, see docs/algorithm.md)
---------------------------------------------------
A pair (i, j) — neither already first order — is coupled when the **wild-type**
network cannot simultaneously supply both target requirements, even though it can
supply either alone. Concretely, build the WT model and test feasibility of:

* ``MUSTUU``: ``v_i >= min_m_i`` AND ``v_j >= min_m_j``
* ``MUSTLL``: ``v_i <= max_m_i`` AND ``v_j <= max_m_j``
* ``MUSTUL``: ``v_i >= min_m_i`` AND ``v_j <= max_m_j``

If the joint system is infeasible while each single bound alone is feasible, the pair
must jointly change. This LP-feasibility form is the transparent equivalent of the
published bilevel search (:mod:`pyoptforce.bilevel`) for moderate candidate pools.
"""

from __future__ import annotations

import itertools
import math

import cobra
from optlang.symbolics import Zero

from pyoptforce.results import FluxRanges, MustSets

_TOL = 1e-6


def first_order(ranges: FluxRanges, *, tol: float = _TOL) -> MustSets:
    """MUSTU / MUSTL via interval logic (see module docstring for directions)."""
    ms = MustSets()
    for r in ranges.reactions():
        if ranges.min_m[r] > ranges.max_w[r] + tol:
            ms.mustU.append(r)
        elif ranges.max_m[r] < ranges.min_w[r] - tol:
            ms.mustL.append(r)
    return ms


def _wt_feasible_with(model: cobra.Model, bounds: dict[str, tuple[str, float]]) -> bool:
    """Is ``model`` feasible when extra single-sided bounds are imposed?

    ``bounds`` maps reaction id -> (``">="`` | ``"<="``, value). Restores bounds on exit.

    A **constant** objective is used so the answer is a pure feasibility test: it never
    depends on the model's objective and an unbounded objective cannot masquerade as
    infeasibility (``slim_optimize`` would return ``nan`` on an unbounded maximise).
    """
    with model:
        model.objective = model.problem.Objective(Zero, direction="max")
        for rid, (sense, val) in bounds.items():
            rxn = model.reactions.get_by_id(rid)
            if sense == ">=":
                rxn.lower_bound = max(rxn.lower_bound, val)
            else:
                rxn.upper_bound = min(rxn.upper_bound, val)
        # NB: slim_optimize(error_value=None) does NOT return None on failure — per its
        # own docstring, error_value=None means "raise instead". Use the default NaN
        # sentinel and check for it, which is the pattern cobra actually supports. This
        # matters here specifically: an infeasible joint bound is the very condition
        # MUSTUU/LL/UL are testing for, so it must be reported as False, not raised.
        sol = model.slim_optimize()
    return not math.isnan(sol)


def second_order(
    model: cobra.Model,
    ranges: FluxRanges,
    must: MustSets,
    *,
    candidates: list[str] | None = None,
    max_pairs: int | None = None,
) -> MustSets:
    """MUSTUU / MUSTLL / MUSTUL by joint WT-feasibility tests.

    ``model`` is the **wild-type** model. Reactions already in first-order sets are
    excluded. ``candidates`` restricts the pool (defaults to all reactions with a
    finite, non-degenerate range); ``max_pairs`` caps the combinatorial search.

    Returns a *new* :class:`MustSets` carrying the existing first-order results plus
    the discovered pairs.
    """
    out = MustSets(mustU=list(must.mustU), mustL=list(must.mustL))
    excluded = set(must.mustU) | set(must.mustL)

    if candidates is None:
        candidates = [
            r for r in ranges.reactions()
            if r not in excluded
            and (ranges.max_w[r] - ranges.min_w[r]) > _TOL
        ]
    else:
        candidates = [r for r in candidates if r not in excluded]

    n = 0
    for i, j in itertools.combinations(candidates, 2):
        if max_pairs is not None and n >= max_pairs:
            break
        n += 1

        # MUSTUU: both forced up to their target minima
        uu = {i: (">=", ranges.min_m[i]), j: (">=", ranges.min_m[j])}
        if not _wt_feasible_with(model, uu) and \
                _wt_feasible_with(model, {i: uu[i]}) and \
                _wt_feasible_with(model, {j: uu[j]}):
            out.mustUU.append((i, j))
            continue

        # MUSTLL: both forced down to their target maxima
        ll = {i: ("<=", ranges.max_m[i]), j: ("<=", ranges.max_m[j])}
        if not _wt_feasible_with(model, ll) and \
                _wt_feasible_with(model, {i: ll[i]}) and \
                _wt_feasible_with(model, {j: ll[j]}):
            out.mustLL.append((i, j))
            continue

        # MUSTUL: i up, j down (and the symmetric i down, j up)
        ul = {i: (">=", ranges.min_m[i]), j: ("<=", ranges.max_m[j])}
        lu = {i: ("<=", ranges.max_m[i]), j: (">=", ranges.min_m[j])}
        if not _wt_feasible_with(model, ul) and \
                _wt_feasible_with(model, {i: ul[i]}) and \
                _wt_feasible_with(model, {j: ul[j]}):
            out.mustUL.append((i, j))
        elif not _wt_feasible_with(model, lu) and \
                _wt_feasible_with(model, {i: lu[i]}) and \
                _wt_feasible_with(model, {j: lu[j]}):
            out.mustUL.append((j, i))

    return out


def find_must_sets(
    model: cobra.Model,
    ranges: FluxRanges,
    *,
    max_order: int = 2,
    max_pairs: int | None = None,
) -> MustSets:
    """Driver: first order, then (if ``max_order >= 2``) second order.

    ``model`` is the wild-type model (needed for the second-order feasibility tests).
    """
    must = first_order(ranges)
    if max_order >= 3:
        raise NotImplementedError(
            "Third-order MUST sets are not implemented (combinatorially expensive); "
            "use max_order<=2."
        )
    if max_order >= 2:
        must = second_order(model, ranges, must, max_pairs=max_pairs)
    return must
