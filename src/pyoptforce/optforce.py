"""Stage 4 + high-level driver.

OptForce class ties stages 1-4 together while keeping every intermediate result
accessible as an attribute (this is what makes the procedure extensible).
"""

from __future__ import annotations

import cobra

from pyoptforce.results import FluxRanges, MustSets, ForceSets


class OptForce:
    def __init__(
        self,
        model: cobra.Model,
        target_reaction: str,
        biomass_reaction: str,
        target_fraction: float = 0.5,
        solver: str = "gurobi",
    ) -> None:
        self.model = model
        self.target_reaction = target_reaction
        self.biomass_reaction = biomass_reaction
        self.target_fraction = target_fraction
        self.solver = solver

        # populated as the pipeline runs — inspect freely
        self.flux_ranges: FluxRanges | None = None
        self.must_sets: MustSets | None = None
        self.force_sets: ForceSets | None = None

    def compute_flux_ranges(self) -> FluxRanges:
        """Stages 1 & 2."""
        raise NotImplementedError

    def find_must_sets(self, max_order: int = 2) -> MustSets:
        """Stage 3."""
        raise NotImplementedError

    def find_force_sets(self, k: int = 1, n_solutions: int = 1) -> ForceSets:
        """Stage 4: the FORCE-set MILP.

        Find minimal sets of <= k interventions that force overproduction for ALL
        feasible mutant flux distributions (the inner adversary, via bilevel.py).
        Enumerate up to n_solutions alternative optima with integer cuts.
        """
        raise NotImplementedError
