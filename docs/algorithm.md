# OptForce: algorithm and bilevel-to-single-level derivation

This document holds the detailed mathematics so the code can reference equation
numbers. Fill in as implementation proceeds.

## Notation
- S: stoichiometric matrix; v: flux vector; vmin/vmax: bounds.
- WT model -> ranges [minFluxesW, maxFluxesW] (stage 1 FVA).
- Target model (constrained to >= target yield) -> [minFluxesM, maxFluxesM] (stage 2).

## Stage 3 — MUST sets
- First order: interval comparison (MUSTU, MUSTL). State exact inequalities here.
- Higher order: bilevel MILP. Outer maximises deviation; inner enforces Sv = 0 and
  bounds. Reduce via strong duality of the inner LP.

## Bilevel -> single-level (the keystone)
1. Write the inner LP in standard form.
2. Form its dual.
3. Strong duality: primal objective == dual objective at optimality.
4. Add that equality + primal feasibility + dual feasibility as constraints to the
   outer MILP. Binary intervention variables gate reactions via indicator/big-M.
5. Choose every big-M from FVA bounds (see bilevel.big_m_from_ranges) and document it.

## Stage 4 — FORCE set
Minimal set of <= k interventions forcing overproduction for ALL feasible mutant
flux distributions. Same bilevel machinery; enumerate alternative optima with
integer cuts.

## Validation targets
- Toy network with hand-computed MUST/FORCE sets (tests/).
- E. coli succinate case study (examples/ecoli_succinate.py) vs published results.
