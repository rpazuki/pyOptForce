"""Bilevel engine — strong-duality reduction of the inner stoichiometric LP.

OptForce's outer problem chooses interventions; the inner problem is an adversarial
flux distribution constrained by stoichiometry. The keystone is reducing that bilevel
structure to a single level using **strong duality of the inner LP**. This module
owns that reduction and is unit-tested IN ISOLATION (see
``tests/test_bilevel.py::test_strong_duality_*``) before being relied on elsewhere.

Inner primal LP (the adversary), for a fixed objective vector ``c`` over fluxes ``v``::

    minimise   cᵀ v
    subject to S v = 0          (λ free)        -- mass balance
               v ≥ lb           (α ≥ 0)         -- lower bounds
               v ≤ ub           (β ≥ 0)         -- upper bounds

Dual::

    maximise   αᵀ lb − βᵀ ub
    subject to Sᵀ λ + α − β = c
               α ≥ 0, β ≥ 0,  λ free

Strong duality at optimality::

    cᵀ v  ==  αᵀ lb − βᵀ ub

Adding the dual feasibility constraints **plus** this equality to an outer model pins
the inner ``v`` to its adversarial optimum, turning max–min into a single max. When an
intervention changes a bound by a constant gated by a binary ``y`` the resulting
bilinear term ``β·(Δub)·y`` is the only nonlinearity. Two ways to remove it:

* **indicator constraints** (Gurobi/CPLEX) — exact, no magic constant. Used by
  :func:`solve_force_milp`.
* **big-M** — portable to any MILP backend, with the constant taken from FVA bounds
  via :func:`big_m_from_ranges` (never an arbitrary value).

See docs/algorithm.md for the full derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyoptforce import solvers


def _finite_or_raise(
    finite_bounds: dict[str, tuple[float, float]] | None,
    rid: str,
    idx: int,
    which: str,
) -> float:
    """Return a finite surrogate for an infinite model bound, or raise.

    ``finite_bounds`` are the WT FVA ranges ``{rid: (min, max)}``. ``idx`` selects
    min (0) or max (1).
    """
    if finite_bounds is not None and rid in finite_bounds:
        val = finite_bounds[rid][idx]
        if math.isfinite(val):
            return float(val)
    raise ValueError(
        f"Reaction {rid!r} has a non-finite {which} bound and no finite surrogate from "
        "FVA; cannot build the strong-duality MILP. Run compute_flux_ranges first or "
        "set finite reaction bounds."
    )


def big_m_from_ranges(min_flux: float, max_flux: float, *, buffer: float = 1.0) -> float:
    """Derive a valid big-M from FVA bounds (never an arbitrary constant).

    A reaction's flux is confined to ``[min_flux, max_flux]``; the tightest constant
    that can dominate any flux magnitude in a gated constraint is therefore
    ``max(|min_flux|, |max_flux|)``. ``buffer`` (additive) guards against equality
    being clipped by solver tolerances. Document every call with the ranges used.
    """
    m = max(abs(min_flux), abs(max_flux))
    return m + buffer


@dataclass
class InnerDual:
    """The dual-variable bundle for one inner stoichiometric LP.

    Attributes mirror the math in the module docstring:
      - ``lam``  : dict metabolite_id -> λ (free)        for ``S v = 0``
      - ``alpha``: dict reaction_id  -> α ≥ 0            for ``v ≥ lb``
      - ``beta`` : dict reaction_id  -> β ≥ 0            for ``v ≤ ub``
      - ``feas`` : the dual-feasibility constraints ``Sᵀλ + α − β = c``
      - ``dual_objective``: the optlang expression ``αᵀ lb − βᵀ ub``
    """

    lam: dict
    alpha: dict
    beta: dict
    feas: list
    dual_objective: object


def build_inner_dual(
    model,
    objective: dict[str, float],
    *,
    interface=None,
    prefix: str = "",
) -> InnerDual:
    """Construct the dual variables/constraints of the inner LP for ``objective``.

    ``objective`` maps reaction id -> coefficient ``c`` for the inner primal
    ``min cᵀv``. ``lb``/``ub`` are taken from each reaction's current bounds. The
    returned :class:`InnerDual` exposes everything needed to assert strong duality.

    Note: this builds the *symbolic* dual on an ``interface`` (optlang) model that the
    caller owns; it does not add constraints to ``model`` itself.
    """
    if interface is None:
        interface = solvers.get_interface()

    lam: dict = {}
    for met in model.metabolites:
        lam[met.id] = interface.Variable(
            f"{prefix}lam_{met.id}", lb=-1e9, ub=1e9
        )

    alpha: dict = {}
    beta: dict = {}
    for rxn in model.reactions:
        alpha[rxn.id] = interface.Variable(f"{prefix}alpha_{rxn.id}", lb=0)
        beta[rxn.id] = interface.Variable(f"{prefix}beta_{rxn.id}", lb=0)

    # Dual feasibility: for each reaction j,  Σ_i S_ij λ_i + α_j − β_j = c_j
    feas: list = []
    for rxn in model.reactions:
        expr = alpha[rxn.id] - beta[rxn.id]
        for met, coeff in rxn.metabolites.items():
            expr = expr + coeff * lam[met.id]
        c_j = objective.get(rxn.id, 0.0)
        con = interface.Constraint(expr, lb=c_j, ub=c_j, name=f"{prefix}dfeas_{rxn.id}")
        feas.append(con)

    # Dual objective: Σ_j α_j·lb_j − β_j·ub_j
    dual_obj = 0
    for rxn in model.reactions:
        dual_obj = dual_obj + alpha[rxn.id] * rxn.lower_bound
        dual_obj = dual_obj - beta[rxn.id] * rxn.upper_bound

    return InnerDual(lam=lam, alpha=alpha, beta=beta, feas=feas, dual_objective=dual_obj)


def inner_primal_value(model, objective: dict[str, float]) -> float:
    """Solve the inner primal ``min cᵀv`` directly (for verification / FORCE step).

    Returns the optimal objective value. Raises on non-optimal status (no silent
    solver failures). Leaves ``model`` unchanged.
    """
    with model:
        # build the linear objective from coefficients
        obj_expr = 0
        for rid, coeff in objective.items():
            obj_expr = obj_expr + coeff * model.reactions.get_by_id(rid).flux_expression
        model.objective = model.problem.Objective(obj_expr, direction="min")
        sol = model.optimize(objective_sense="minimize")
        if sol.status != "optimal":
            raise RuntimeError(f"Inner primal LP not optimal (status={sol.status!r}).")
        return float(sol.objective_value)


def strong_duality_selftest(interface=None) -> tuple[float, float]:
    """Isolation test of the reduction on a trivial LP (roadmap step 3).

    Problem::

        min  v1 + v2
        s.t. v1 − v2 = 0     (one 'metabolite')
             0 ≤ v1 ≤ 5
             1 ≤ v2 ≤ 4

    Optimal primal = 1 (v1=v2=1). We build the dual via the same algebra used by
    :func:`build_inner_dual` and check the dual optimum equals the primal optimum.
    Returns ``(primal_opt, dual_opt)``.
    """
    if interface is None:
        interface = solvers.get_interface()

    # ---- primal ----
    v1 = interface.Variable("v1", lb=0, ub=5)
    v2 = interface.Variable("v2", lb=1, ub=4)
    balance = interface.Constraint(v1 - v2, lb=0, ub=0, name="balance")
    pm = interface.Model(name="primal")
    pm.add([v1, v2, balance])
    pm.objective = interface.Objective(v1 + v2, direction="min")
    pm.optimize()
    primal_opt = float(pm.objective.value)

    # ---- dual (built by hand, mirroring build_inner_dual) ----
    # S = [[1, -1]] (single row). c = [1, 1]. lb=[0,1], ub=[5,4].
    lam = interface.Variable("lam", lb=-1e9, ub=1e9)
    a1 = interface.Variable("a1", lb=0)
    a2 = interface.Variable("a2", lb=0)
    b1 = interface.Variable("b1", lb=0)
    b2 = interface.Variable("b2", lb=0)
    # Sᵀλ + α − β = c   ->  v1:  1·lam + a1 − b1 = 1 ;  v2: −1·lam + a2 − b2 = 1
    f1 = interface.Constraint(lam + a1 - b1, lb=1, ub=1, name="df1")
    f2 = interface.Constraint(-lam + a2 - b2, lb=1, ub=1, name="df2")
    dm = interface.Model(name="dual")
    dm.add([lam, a1, a2, b1, b2, f1, f2])
    # αᵀlb − βᵀub = 0·a1 + 1·a2 − 5·b1 − 4·b2
    dm.objective = interface.Objective(a2 - 5 * b1 - 4 * b2, direction="max")
    dm.optimize()
    dual_opt = float(dm.objective.value)

    if not np.isclose(primal_opt, dual_opt, atol=1e-6):
        raise AssertionError(
            f"Strong duality failed: primal={primal_opt}, dual={dual_opt}"
        )
    return primal_opt, dual_opt


def solve_force_milp(
    model,
    *,
    target_reaction: str,
    candidates: dict[str, str],
    forced_value: dict[str, float],
    growth_floor: tuple[str, float] | None,
    k: int,
    threshold: float,
    finite_bounds: dict[str, tuple[float, float]] | None = None,
    n_solutions: int = 1,
    tol: float = 1e-6,
) -> list[dict]:
    """Single-level FORCE MILP via strong duality (direct gurobipy, indicator form).

    This is the efficient counterpart of the enumerate-and-verify FORCE search: instead
    of testing every ``<=k`` subset it lets the MILP *choose* the interventions, while
    strong duality of the inner adversary LP guarantees the worst case.

    Ranganathan et al. 2010 — OptForce step (their Eqs. 7-12, the FORCE-set bilevel and
    its single-level dual reduction).

    Decision variables
    -------------------
    * ``v[j]``  flux of reaction j (base WT bounds; biomass floor folded into its lb).
    * ``y[j]``  ∈ {0,1}, intervention on candidate j selected.
    * ``lam[i]`` free, dual of mass balance ``S v = 0``.
    * ``alpha[j] >= 0`` / ``beta[j] >= 0``, duals of the lower / upper flux bounds.
    * ``p[j]`` = ``alpha[j]·y[j]`` (up candidates), ``q[j]`` = ``beta[j]·y[j]`` (down) —
      products linearised with **indicator constraints** (no big-M).

    Constraints
    -----------
    * mass balance:                ``Σ_i S[i,j] v[j] = 0``        (∀ metabolite i)
    * engineered bound (indicator):
        up   candidate: ``y[j]=1 ⇒ v[j] >= forced_value[j]``
        down candidate: ``y[j]=1 ⇒ v[j] <= forced_value[j]``
    * dual feasibility:            ``Σ_i S[i,j] lam[i] + alpha[j] − beta[j] = c[j]``
    * strong duality:              ``cᵀv = Σ alpha·lb_eff(y) − Σ beta·ub_eff(y)``
      with ``lb_eff``/``ub_eff`` expanded through the indicator products p/q.
    * cardinality:                 ``Σ_j y[j] <= k``

    Objective
    ---------
    maximise ``v[target]`` — strong duality pins it to the inner *minimum*, so the MILP
    maximises the worst-case (guaranteed) target flux over the choice of interventions.

    A solution is a valid FORCE set iff its objective ``>= threshold``. Alternative
    optima are enumerated with integer cuts. Returns a list of
    ``{"reactions", "type", "objective"}`` dicts (same shape as the LP path).

    Infinite model bounds (some SBML use ``±inf``) would put an infinite coefficient in
    the dual objective and break strong duality. ``finite_bounds`` (the WT FVA ranges,
    which are the actual reachable flux extremes — interventions only tighten, never
    widen) supplies a valid finite surrogate; an unbacked ``±inf`` bound raises.
    """
    gp = solvers.require_gurobi()
    GRB = gp.GRB

    reactions = list(model.reactions)
    rids = [r.id for r in reactions]
    c = {rid: (1.0 if rid == target_reaction else 0.0) for rid in rids}

    # Base WT bounds, made finite via the FVA ranges where the model uses ±inf, with the
    # viability floor folded into the biomass lower bound.
    lb: dict[str, float] = {}
    ub: dict[str, float] = {}
    for r in reactions:
        lo, hi = float(r.lower_bound), float(r.upper_bound)
        if not math.isfinite(lo):
            lo = _finite_or_raise(finite_bounds, r.id, 0, "lower")
        if not math.isfinite(hi):
            hi = _finite_or_raise(finite_bounds, r.id, 1, "upper")
        lb[r.id], ub[r.id] = lo, hi
    if growth_floor is not None:
        bio_id, floor = growth_floor
        lb[bio_id] = max(lb[bio_id], float(floor))

    cand_ids = list(candidates)

    g = gp.Model("optforce_force")
    g.Params.OutputFlag = 0

    v = {rid: g.addVar(lb=lb[rid], ub=ub[rid], name=f"v_{rid}") for rid in rids}
    lam = {m.id: g.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"lam_{m.id}")
           for m in model.metabolites}
    alpha = {rid: g.addVar(lb=0.0, name=f"a_{rid}") for rid in rids}
    beta = {rid: g.addVar(lb=0.0, name=f"b_{rid}") for rid in rids}
    y = {j: g.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in cand_ids}
    # linearisation auxiliaries for the bilinear alpha·y / beta·y in the dual objective
    p = {j: g.addVar(lb=0.0, name=f"p_{j}") for j in cand_ids if candidates[j] == "up"}
    q = {j: g.addVar(lb=0.0, name=f"q_{j}") for j in cand_ids if candidates[j] == "down"}
    g.update()

    # mass balance  Σ_j S[i,j] v[j] = 0
    for met in model.metabolites:
        g.addConstr(
            gp.quicksum(rxn.get_coefficient(met.id) * v[rxn.id]
                        for rxn in met.reactions) == 0.0,
            name=f"mb_{met.id}",
        )

    # engineered bounds via indicator constraints
    for j, direction in candidates.items():
        if direction == "up":
            g.addGenConstrIndicator(y[j], True, v[j], GRB.GREATER_EQUAL,
                                    forced_value[j], name=f"force_up_{j}")
        else:  # down
            g.addGenConstrIndicator(y[j], True, v[j], GRB.LESS_EQUAL,
                                    forced_value[j], name=f"force_dn_{j}")

    # dual feasibility  Σ_i S[i,j] lam[i] + alpha[j] − beta[j] = c[j]
    for rxn in reactions:
        g.addConstr(
            gp.quicksum(coeff * lam[met.id] for met, coeff in rxn.metabolites.items())
            + alpha[rxn.id] - beta[rxn.id] == c[rxn.id],
            name=f"dfeas_{rxn.id}",
        )

    # indicator linearisation: p_j = alpha_j·y_j , q_j = beta_j·y_j
    for j in p:
        g.addConstr(p[j] <= alpha[j], name=f"p_le_{j}")
        g.addGenConstrIndicator(y[j], False, p[j], GRB.EQUAL, 0.0, name=f"p0_{j}")
        g.addGenConstrIndicator(y[j], True, p[j] - alpha[j], GRB.EQUAL, 0.0,
                                name=f"p1_{j}")
    for j in q:
        g.addConstr(q[j] <= beta[j], name=f"q_le_{j}")
        g.addGenConstrIndicator(y[j], False, q[j], GRB.EQUAL, 0.0, name=f"q0_{j}")
        g.addGenConstrIndicator(y[j], True, q[j] - beta[j], GRB.EQUAL, 0.0,
                                name=f"q1_{j}")

    # strong duality:  cᵀv = Σ alpha·lb_eff − Σ beta·ub_eff
    #   lb_eff_j = lb_j + y_j·(forced − lb_j)  for up candidates -> alpha·lb_eff
    #              = alpha_j·lb_j + (forced−lb_j)·p_j
    #   ub_eff_j = ub_j + y_j·(forced − ub_j)  for down candidates -> beta·ub_eff
    #              = beta_j·ub_j + (forced−ub_j)·q_j
    dual_lower = gp.quicksum(alpha[rid] * lb[rid] for rid in rids) \
        + gp.quicksum((forced_value[j] - lb[j]) * p[j] for j in p)
    dual_upper = gp.quicksum(beta[rid] * ub[rid] for rid in rids) \
        + gp.quicksum((forced_value[j] - ub[j]) * q[j] for j in q)
    primal_obj = gp.quicksum(c[rid] * v[rid] for rid in rids)
    g.addConstr(primal_obj == dual_lower - dual_upper, name="strong_duality")

    # cardinality
    g.addConstr(gp.quicksum(y[j] for j in cand_ids) <= k, name="cardinality")

    g.setObjective(v[target_reaction], GRB.MAXIMIZE)

    solutions: list[dict] = []
    for _ in range(n_solutions):
        g.optimize()
        if g.Status != GRB.OPTIMAL:
            break
        worst = float(g.ObjVal)
        if worst < threshold - tol:
            break
        chosen = [j for j in cand_ids if y[j].X > 0.5]
        solutions.append({
            "reactions": chosen,
            "type": {j: candidates[j] for j in chosen},
            "objective": worst,
        })
        # integer cut: forbid exactly this selection, find the next optimum
        sel = set(chosen)
        g.addConstr(
            gp.quicksum(1 - y[j] for j in sel)
            + gp.quicksum(y[j] for j in cand_ids if j not in sel) >= 1,
            name=f"cut_{len(solutions)}",
        )
    return solutions
