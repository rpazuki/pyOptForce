"""Stages 1 & 2: flux variability analysis for WT and target models."""

from __future__ import annotations

import cobra

from pyoptforce.results import FluxRanges


def wild_type_ranges(model: cobra.Model, *, fraction_of_optimum: float = 0.0) -> dict:
    """Stage 1: FVA on the wild-type model -> minFluxesW, maxFluxesW.

    Thin wrapper over cobra.flux_analysis.flux_variability_analysis with the
    OptForce-appropriate defaults. Returns {rxn_id: (min, max)}.
    """
    raise NotImplementedError


def target_ranges(target_model: cobra.Model) -> dict:
    """Stage 2: FVA on the target-constrained model -> minFluxesM, maxFluxesM."""
    raise NotImplementedError


def compute_flux_ranges(model: cobra.Model, target_model: cobra.Model) -> FluxRanges:
    """Run both FVA stages and pack into a FluxRanges container."""
    raise NotImplementedError
