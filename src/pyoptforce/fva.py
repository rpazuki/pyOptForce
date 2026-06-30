"""Stages 1 & 2: flux variability analysis for WT and target models."""

from __future__ import annotations

import cobra
from cobra.flux_analysis import flux_variability_analysis

from pyoptforce.results import FluxRanges


def _fva_dict(
    model: cobra.Model,
    *,
    fraction_of_optimum: float,
    reaction_list=None,
) -> dict[str, tuple[float, float]]:
    """Run cobra FVA and return ``{rxn_id: (min, max)}``."""
    df = flux_variability_analysis(
        model,
        reaction_list=reaction_list,
        fraction_of_optimum=fraction_of_optimum,
    )
    return {rid: (float(row["minimum"]), float(row["maximum"]))
            for rid, row in df.iterrows()}


def wild_type_ranges(
    model: cobra.Model, *, fraction_of_optimum: float = 0.0
) -> dict[str, tuple[float, float]]:
    """Stage 1: FVA on the wild-type model -> minFluxesW, maxFluxesW.

    ``fraction_of_optimum`` constrains the model's **current objective**; the OptForce
    driver anchors that to the biomass reaction before calling, so the WT basal state is
    measured against growth (not whatever objective the SBML shipped). Standalone callers
    must set the intended objective themselves. ``fraction_of_optimum=0.0`` explores the
    full feasible space. Returns ``{rxn_id: (min, max)}``.
    """
    return _fva_dict(model, fraction_of_optimum=fraction_of_optimum)


def target_ranges(target_model: cobra.Model) -> dict[str, tuple[float, float]]:
    """Stage 2: FVA on the target-constrained ('M') model -> minFluxesM, maxFluxesM.

    The target model already carries the overproduction lower bound (see
    :func:`pyoptforce.model.set_target_yield`), so FVA here is run with
    ``fraction_of_optimum=0.0``: we want the full range of fluxes *consistent with the
    target*, not those tied to maximal biomass.
    """
    return _fva_dict(target_model, fraction_of_optimum=0.0)


def compute_flux_ranges(
    model: cobra.Model,
    target_model: cobra.Model,
    *,
    fraction_of_optimum: float = 0.0,
) -> FluxRanges:
    """Run both FVA stages and pack into a :class:`FluxRanges` container."""
    w = wild_type_ranges(model, fraction_of_optimum=fraction_of_optimum)
    m = target_ranges(target_model)

    fr = FluxRanges()
    for rid, (lo, hi) in w.items():
        fr.min_w[rid] = lo
        fr.max_w[rid] = hi
    for rid, (lo, hi) in m.items():
        fr.min_m[rid] = lo
        fr.max_m[rid] = hi
    return fr
