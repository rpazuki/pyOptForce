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
"""

from __future__ import annotations

import cobra


def prepare_model(model: cobra.Model, *, copy: bool = True) -> cobra.Model:
    """Return a model ready for OptForce.

    - deep-copies by default so the caller's model is never mutated;
    - sanity-checks that the model has reactions and a feasible objective.

    Reversibility is left signed (see module docstring); no reaction splitting.
    """
    prepared = model.copy() if copy else model

    if len(prepared.reactions) == 0:
        raise ValueError("Model has no reactions.")

    sol = prepared.optimize()
    if sol.status != "optimal":
        raise ValueError(
            f"Base model is not optimisable (status={sol.status!r}); "
            "fix bounds/objective before running OptForce."
        )
    return prepared


def theoretical_max(model: cobra.Model, target_reaction: str) -> float:
    """Maximum flux through ``target_reaction`` over the model's feasible space.

    Used to anchor the target yield. Leaves the input model untouched.
    """
    with model:
        model.objective = target_reaction
        model.objective_direction = "max"
        sol = model.optimize()
        if sol.status != "optimal":
            raise ValueError(
                f"Could not maximise {target_reaction!r} (status={sol.status!r})."
            )
        return float(sol.objective_value)


def set_target_yield(
    model: cobra.Model,
    target_reaction: str,
    biomass_reaction: str,
    target_fraction: float,
    *,
    copy: bool = True,
) -> cobra.Model:
    """Constrain the model to the overproducing ('M') phenotype.

    Recipe: maximise ``target_reaction`` to get its theoretical maximum, then fix its
    **lower bound** at ``target_fraction * max`` so every feasible flux distribution
    must overproduce at least that much. The objective is left on the biomass reaction
    so the model still represents a growing strain. The returned model is the one
    stage-2 FVA explores.
    """
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must be in [0, 1].")

    m = model.copy() if copy else model
    vmax = theoretical_max(m, target_reaction)
    threshold = target_fraction * vmax

    rxn = m.reactions.get_by_id(target_reaction)
    # Force overproduction: target flux >= threshold for all feasible distributions.
    rxn.lower_bound = max(rxn.lower_bound, threshold)

    # Keep the biomass reaction as the cellular objective (growing strain).
    m.objective = biomass_reaction
    m.objective_direction = "max"
    return m
