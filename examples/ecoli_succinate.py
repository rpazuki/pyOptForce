"""End-to-end run: E. coli succinate overproduction (canonical OptForce case).

Uses the small ``e_coli_core`` model for a fast first pass (swap in ``iJO1366`` for the
full case study). Requires network access the first time to download the model, and an
installed MILP/LP backend (Gurobi/CPLEX/SCIP/GLPK — discovered automatically).

Run::

    python examples/ecoli_succinate.py
"""

from __future__ import annotations

import cobra

from pyoptforce import OptForce


def main() -> None:
    model = cobra.io.load_model("e_coli_core")

    of = OptForce(
        model,
        target_reaction="EX_succ_e",
        biomass_reaction="BIOMASS_Ecoli_core_w_GAM",
        target_fraction=0.3,
        solver="auto",
    )

    of.compute_flux_ranges()
    print("flux ranges (head):")
    print(of.flux_ranges.to_dataframe().head())

    must = of.find_must_sets(max_order=1)  # max_order=2 is slower (pairwise LPs)
    print("\nMUST sets:")
    print(must.to_dataframe())

    force = of.find_force_sets(k=2, n_solutions=10)
    print("\nFORCE sets:")
    print(force.to_dataframe())


if __name__ == "__main__":
    main()
