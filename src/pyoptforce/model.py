"""Model preparation: loading, target setup, and reversibility handling.

Stage 0 of the pipeline. Everything downstream assumes the conventions set here
(split vs signed reversible reactions, where the target/biomass live, etc.).
"""

from __future__ import annotations

import cobra


def prepare_model(model: cobra.Model, *, copy: bool = True) -> cobra.Model:
    """Return a model ready for OptForce.

    Responsibilities (to implement):
      - optionally deep-copy so we never mutate the caller's model
      - normalise reversibility handling (decide: split into fwd/rev, or keep signed)
      - sanity-check bounds and objective
    """
    raise NotImplementedError


def set_target_yield(
    model: cobra.Model,
    target_reaction: str,
    biomass_reaction: str,
    target_fraction: float,
) -> cobra.Model:
    """Constrain the model to the overproducing phenotype.

    Typical recipe: compute the theoretical max of `target_reaction`, then fix a
    lower bound at `target_fraction * max`, producing the 'M' (mutant/target) model
    used for stage-2 FVA. Implement and document the exact constraint choice here.
    """
    raise NotImplementedError
