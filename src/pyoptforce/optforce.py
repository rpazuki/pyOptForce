"""Stage 4 + high-level driver.

OptForce class ties stages 1-4 together while keeping every intermediate result
accessible as an attribute (this is what makes the procedure extensible).

FORCE step (stage 4)
--------------------
Given the MUST sets, an intervention assigns each candidate reaction a *direction*
inherited from its MUST membership (MUSTU -> "up", MUSTL -> "down"; pair members take
their pair's direction). Applying an intervention tightens the engineered strain's
bounds:

* "up":   ``lower_bound := min_m``  (force at least the target minimum flux)
* "down": ``upper_bound := max_m``  (cap at the target maximum flux)

A candidate set is a valid FORCE set when, on the engineered model (wild-type bounds +
interventions + a viability floor on biomass), the **worst-case** target flux is still
>= the overproduction threshold. The worst case is the inner adversary
``min target_flux``; because it is a plain LP we solve it directly (exact and
inspectable) rather than dualising into one monolithic MILP — the bilevel dual machinery
in :mod:`pyoptforce.bilevel` backs the same guarantee and is unit-tested in isolation.

Sets are enumerated smallest-first over subsets of size <= k; up to ``n_solutions``
minimal valid sets are returned (the analogue of integer-cut enumeration).
"""

from __future__ import annotations

import itertools

import cobra

from pyoptforce import fva, model as model_mod, must_sets as ms_mod
from pyoptforce.results import FluxRanges, MustSets, ForceSets

_TOL = 1e-6


class OptForce:
    def __init__(
        self,
        model: cobra.Model,
        target_reaction: str,
        biomass_reaction: str,
        target_fraction: float = 0.5,
        solver: str = "auto",
        *,
        min_biomass_fraction: float = 0.1,
        wt_growth_fraction: float = 1.0,
    ) -> None:
        self.model = model_mod.prepare_model(model, copy=True)
        # Validate ids up front against THIS organism's SBML (clear error, not a deep
        # KeyError later) and forbid the degenerate target==biomass case.
        model_mod.require_reactions(self.model, [target_reaction, biomass_reaction])
        if target_reaction == biomass_reaction:
            raise ValueError(
                "target_reaction and biomass_reaction must differ "
                f"(both {target_reaction!r})."
            )

        self.target_reaction = target_reaction
        self.biomass_reaction = biomass_reaction
        self.target_fraction = target_fraction
        self.solver = solver
        if not 0.0 <= min_biomass_fraction <= 1.0:
            raise ValueError("min_biomass_fraction must be in [0, 1].")
        if not 0.0 <= wt_growth_fraction <= 1.0:
            raise ValueError("wt_growth_fraction must be in [0, 1].")
        self.min_biomass_fraction = min_biomass_fraction
        self.wt_growth_fraction = wt_growth_fraction

        # point cobra at an installed backend
        from pyoptforce import solvers
        self.model.solver = solvers.cobra_solver_name(solver)

        # Anchor the cellular objective to the biomass reaction explicitly and once, so
        # no stage ever inherits whatever objective the SBML happened to define.
        model_mod.set_linear_objective(self.model, self.biomass_reaction, "max")

        # populated as the pipeline runs — inspect freely
        self.target_model: cobra.Model | None = None
        self.flux_ranges: FluxRanges | None = None
        self.must_sets: MustSets | None = None
        self.force_sets: ForceSets | None = None
        self.target_threshold: float | None = None
        self.max_growth: float | None = None

    # ------------------------------------------------------------------ stages 1&2
    def compute_flux_ranges(self) -> FluxRanges:
        """Stages 1 & 2: WT FVA and target-constrained FVA."""
        # Re-assert the biomass objective defensively (the caller may have touched
        # self.model since construction): the WT basal state, stage-1 FVA
        # fraction_of_optimum, and max_growth must all be measured against biomass —
        # never whatever objective the SBML happened to define.
        model_mod.set_linear_objective(self.model, self.biomass_reaction, "max")

        self.max_growth = model_mod.theoretical_max(self.model, self.biomass_reaction)
        if self.max_growth <= _TOL:
            raise ValueError(
                f"Wild-type cannot grow (max biomass={self.max_growth:.3g}); OptForce "
                "assumes a viable growing strain. Check the biomass reaction and uptake "
                "bounds."
            )

        vmax = model_mod.theoretical_max(self.model, self.target_reaction)
        if vmax <= _TOL:
            raise ValueError(
                f"Target {self.target_reaction!r} cannot be produced "
                f"(max={vmax:.3g}). Check the reaction id, orientation, and uptake."
            )
        self.target_threshold = self.target_fraction * vmax

        self.target_model = model_mod.set_target_yield(
            self.model,
            self.target_reaction,
            self.biomass_reaction,
            self.target_fraction,
            copy=True,
        )
        # WT basal state = wild type at (a fraction of) maximum growth. Without a
        # growth floor the target space is a strict subset of the WT space and every
        # MUST test is vacuous, so the default takes the WT at maximum growth.
        self.flux_ranges = fva.compute_flux_ranges(
            self.model, self.target_model,
            fraction_of_optimum=self.wt_growth_fraction,
        )
        return self.flux_ranges

    # -------------------------------------------------------------------- stage 3
    def find_must_sets(self, max_order: int = 2, *, max_pairs: int | None = None) -> MustSets:
        """Stage 3: MUST-set classification."""
        if self.flux_ranges is None:
            self.compute_flux_ranges()
        self.must_sets = ms_mod.find_must_sets(
            self.model, self.flux_ranges, max_order=max_order, max_pairs=max_pairs
        )
        return self.must_sets

    # -------------------------------------------------------------------- stage 4
    def _candidate_directions(self) -> dict[str, str]:
        """Map each MUST reaction to its intervention direction ("up"/"down").

        First-order memberships win over pair-derived directions on conflict.
        """
        ms = self.must_sets
        directions: dict[str, str] = {}
        for (i, j) in ms.mustUU:
            directions.setdefault(i, "up")
            directions.setdefault(j, "up")
        for (i, j) in ms.mustLL:
            directions.setdefault(i, "down")
            directions.setdefault(j, "down")
        for (i, j) in ms.mustUL:  # stored as (up-reaction, down-reaction)
            directions.setdefault(i, "up")
            directions.setdefault(j, "down")
        for r in ms.mustU:
            directions[r] = "up"
        for r in ms.mustL:
            directions[r] = "down"
        return directions

    def _worst_case_target(self, interventions: dict[str, str]) -> float | None:
        """Inner adversary: min target flux on the engineered viable strain.

        Returns the worst-case target flux, or ``None`` if the engineered strain is
        infeasible (cannot even grow at the viability floor).
        """
        fr = self.flux_ranges
        min_growth = self.min_biomass_fraction * self.max_growth
        with self.model as m:
            m.reactions.get_by_id(self.biomass_reaction).lower_bound = min_growth
            for rid, direction in interventions.items():
                rxn = m.reactions.get_by_id(rid)
                if direction == "up":
                    rxn.lower_bound = max(rxn.lower_bound, fr.min_m[rid])
                else:  # down
                    rxn.upper_bound = min(rxn.upper_bound, fr.max_m[rid])
            m.objective = self.target_reaction
            m.objective_direction = "min"  # adversary minimises the target
            val = m.slim_optimize(error_value=None)
        return None if val is None else float(val)

    def _force_sets_milp(
        self, directions: dict[str, str], k: int, n_solutions: int
    ) -> ForceSets:
        """FORCE sets via the single-level strong-duality MILP (Gurobi)."""
        from pyoptforce import bilevel

        fr = self.flux_ranges
        forced = {
            r: (fr.min_m[r] if d == "up" else fr.max_m[r])
            for r, d in directions.items()
        }
        growth_floor = (self.biomass_reaction,
                        self.min_biomass_fraction * self.max_growth)
        # WT FVA ranges = finite surrogates for any ±inf model bounds in the dual.
        finite_bounds = {
            r: (fr.min_w[r], fr.max_w[r]) for r in fr.min_w
        }
        sets = bilevel.solve_force_milp(
            self.model,
            target_reaction=self.target_reaction,
            candidates=directions,
            forced_value=forced,
            growth_floor=growth_floor,
            k=k,
            threshold=self.target_threshold,
            finite_bounds=finite_bounds,
            n_solutions=n_solutions,
            tol=_TOL,
        )
        return ForceSets(sets=sets)

    def find_force_sets(
        self, k: int = 1, n_solutions: int = 1, *, method: str = "auto"
    ) -> ForceSets:
        """Stage 4: find minimal FORCE sets of size <= ``k``.

        ``method``:
          - ``"milp"``      single-level strong-duality MILP via direct gurobipy
            (efficient: the solver *chooses* interventions). Requires Gurobi.
          - ``"enumerate"`` exact enumerate-and-verify over subsets, LP-only (any
            backend). Slower for large candidate pools but solver-agnostic.
          - ``"auto"``      MILP if Gurobi is installed, else enumerate.

        Returns up to ``n_solutions`` minimal valid sets, smallest-first.
        """
        from pyoptforce import solvers

        if self.must_sets is None:
            self.find_must_sets()
        if self.target_threshold is None:
            self.compute_flux_ranges()

        directions = self._candidate_directions()
        threshold = self.target_threshold

        if method == "auto":
            method = "milp" if solvers.gurobi_available() else "enumerate"
        if method == "milp":
            self.force_sets = self._force_sets_milp(directions, k, n_solutions)
            return self.force_sets
        if method != "enumerate":
            raise ValueError(f"Unknown method {method!r}; use auto/milp/enumerate.")

        candidates = list(directions)
        found: list[dict] = []
        seen: set[frozenset[str]] = set()

        for size in range(1, k + 1):
            for combo in itertools.combinations(candidates, size):
                key = frozenset(combo)
                # skip supersets of an already-found minimal set
                if any(prev <= key for prev in seen):
                    continue
                interventions = {r: directions[r] for r in combo}
                worst = self._worst_case_target(interventions)
                if worst is not None and worst >= threshold - _TOL:
                    found.append({
                        "reactions": list(combo),
                        "type": interventions,
                        "objective": worst,
                    })
                    seen.add(key)
                    if len(found) >= n_solutions:
                        self.force_sets = ForceSets(sets=found)
                        return self.force_sets
        self.force_sets = ForceSets(sets=found)
        return self.force_sets
