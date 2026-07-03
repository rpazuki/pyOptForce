"""Model preparation: loading, target setup, and reversibility handling.

Stage 0 of the pipeline. Everything downstream assumes the conventions set here.

Reversibility convention
------------------------
We keep reversible reactions **signed** (a single variable with ``lower_bound < 0``)
rather than splitting them into forward/reverse pairs. cobra's FVA, the stoichiometric
matrix ``S`` and the optlang variables all use this convention, so keeping it avoids a
translation layer and keeps reaction ids stable across every stage. The bilevel
reduction therefore handles reactions that can carry flux in either direction directly
through their (possibly negative) bounds.

Objective convention
--------------------
Nothing in this module trusts the objective an SBML file happens to ship with. Callers
state which reaction is the cellular objective explicitly (``set_linear_objective``);
feasibility is probed with a *constant* objective so the result never depends on what the
file defined. This keeps the pipeline correct across arbitrary organisms' models.
"""

from __future__ import annotations

import cobra
from optlang.symbolics import Zero

_TOL = 1e-9


def require_reactions(model: cobra.Model, reaction_ids: list[str]) -> None:
    """Raise a clear error if any id is missing (organism/model mismatch).

    cobra would otherwise raise a bare ``KeyError`` deep in a later stage; surface the
    problem up front with the offending ids.
    """
    # cobra DictList lookups are indexed by id; checking only requested ids avoids
    # rebuilding a full-id set on every call.
    for rid in dict.fromkeys(reaction_ids):
        try:
            model.reactions.get_by_id(rid)
        except KeyError:
            raise KeyError(
                f"Reaction id not in model: {rid!r}. "
                f"Check ids against this organism's SBML (the model has "
                f"{len(model.reactions)} reactions)."
            ) from None


def set_linear_objective(
    model: cobra.Model, reaction_id: str, direction: str = "max"
) -> None:
    """Set the model objective to a single reaction, explicitly and direction-safe."""
    require_reactions(model, [reaction_id])
    model.objective = reaction_id
    model.objective_direction = direction


def is_feasible(model: cobra.Model) -> bool:
    """Does the model have *any* feasible flux distribution?

    Uses a constant (zero) objective so the answer is independent of the model's current
    objective — a model whose default objective is unbounded is still feasible.
    """
    with model:
        model.objective = model.problem.Objective(Zero, direction="max")
        value = model.slim_optimize(error_value=None)
    return value is not None


def prepare_model(model: cobra.Model, *, copy: bool = True) -> cobra.Model:
    """Return a model ready for OptForce.

    - deep-copies by default so the caller's model is never mutated;
    - checks it has reactions and is *feasible* (independent of its objective).

    Reversibility is left signed (see module docstring); no reaction splitting.
    """
    prepared = model.copy() if copy else model

    if len(prepared.reactions) == 0:
        raise ValueError("Model has no reactions.")
    if len(prepared.metabolites) == 0:
        raise ValueError("Model has no metabolites.")
    if not is_feasible(prepared):
        raise ValueError(
            "Model is infeasible at its given bounds; fix bounds before running "
            "OptForce."
        )
    return prepared


def theoretical_max(
    model: cobra.Model, reaction_id: str, *, direction: str = "max"
) -> float:
    """Optimal flux through ``reaction_id`` over the feasible space.

    Restores the model's objective and bounds on exit (``with model:``), so the input is
    never mutated. Raises on non-optimal status (e.g. unbounded), never silently.
    """
    require_reactions(model, [reaction_id])
    with model:
        model.objective = reaction_id
        model.objective_direction = direction
        value = model.slim_optimize(error_value=None)
        if value is None:
            status = model.solver.status
            raise ValueError(
                f"Could not optimise {reaction_id!r} (status={status!r}); the flux is "
                "likely unbounded — set finite bounds on the relevant reactions."
            )
        return float(value)


def set_target_yield(
    model: cobra.Model,
    target_reaction: str,
    biomass_reaction: str,
    target_fraction: float,
    *,
    copy: bool = True,
    tol: float = _TOL,
) -> cobra.Model:
    """Constrain the model to the overproducing ('M') phenotype.

    Recipe: maximise ``target_reaction`` to get its theoretical maximum, then fix its
    **lower bound** at ``target_fraction * max`` so every feasible flux distribution must
    overproduce at least that much. The biomass reaction becomes the objective (growing
    strain). The returned model is the one stage-2 FVA explores.

    Guards (organism-agnostic): both reactions must exist, must differ, and the target
    must be producible with positive flux (``vmax > 0``) — otherwise the yield
    constraint is meaningless.
    """
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be in [0, 1].")
    require_reactions(model, [target_reaction, biomass_reaction])
    if target_reaction == biomass_reaction:
        raise ValueError(
            "target_reaction and biomass_reaction must differ "
            f"(both {target_reaction!r})."
        )

    m = model.copy() if copy else model
    vmax = theoretical_max(m, target_reaction)
    if vmax <= tol:
        raise ValueError(
            f"Target {target_reaction!r} cannot carry positive flux (max={vmax:.3g}); "
            "it is not a producible product under the current bounds. Check the "
            "reaction id, its orientation, and substrate uptake bounds."
        )
    threshold = target_fraction * vmax

    rxn = m.reactions.get_by_id(target_reaction)
    if rxn.upper_bound < threshold - tol:
        # cannot happen for a well-posed model (threshold <= vmax <= ub) but guard
        # against bound inconsistencies in third-party SBML.
        raise ValueError(
            f"Target upper bound {rxn.upper_bound:.3g} below required yield "
            f"{threshold:.3g}; the target model would be infeasible."
        )
    # Force overproduction: target flux >= threshold for all feasible distributions.
    rxn.lower_bound = max(rxn.lower_bound, threshold)

    set_linear_objective(m, biomass_reaction, "max")
    if not is_feasible(m):
        raise ValueError(
            f"Target model is infeasible at yield {threshold:.3g}; the organism cannot "
            "sustain this overproduction. Lower target_fraction or relax bounds."
        )
    return m
