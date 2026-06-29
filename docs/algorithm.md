# OptForce: algorithm and bilevel-to-single-level derivation

Detailed mathematics behind each stage, with pointers to the code. We re-derive the
formulations from Ranganathan et al. 2010 rather than translating the GAMS files.

## Notation
- `S`: stoichiometric matrix; `v`: flux vector; `lb`/`ub`: bounds.
- WT model → ranges `[min_w, max_w]` (stage 1 FVA, `fva.wild_type_ranges`).
- Target model (constrained to ≥ target yield) → `[min_m, max_m]`
  (stage 2 FVA, `fva.target_ranges`).

## Stages 1 & 2 — flux ranges
The **wild-type basal state** is the WT network at (a fraction of) maximum growth:
FVA is run with `fraction_of_optimum = wt_growth_fraction` (default 1.0). This matters —
the target model is the WT model plus the overproduction lower bound, so its feasible
set is a *subset* of the unconstrained WT set. Without the growth floor every MUST test
would be vacuous (`min_m ≥ min_w`, `max_m ≤ max_w` always). Pinning the WT at maximum
growth makes the two range sets genuinely diverge.

The **target ('M') model** (`model.set_target_yield`): maximise `target_reaction` to get
its theoretical maximum `vmax`, then fix `lower_bound := target_fraction · vmax` so every
feasible distribution overproduces. The biomass reaction stays the objective (growing
strain). Stage-2 FVA then uses `fraction_of_optimum = 0` to span all flux consistent with
the target.

## Stage 3 — MUST sets

### First order (`must_sets.first_order`) — interval logic
For reaction `j`:
- **MUSTU** (must increase): `min_m_j > max_w_j` — the lowest target flux exceeds the
  highest WT flux.
- **MUSTL** (must decrease): `max_m_j < min_w_j` — the highest target flux is below the
  lowest WT flux.

### Second order (`must_sets.second_order`) — joint WT-feasibility
A pair `(i, j)`, neither already first order, is coupled when the **wild-type** network
cannot meet both target requirements at once though it can meet either alone. Testing
feasibility of the WT model under added single-sided bounds:
- **MUSTUU**: `v_i ≥ min_m_i` ∧ `v_j ≥ min_m_j`
- **MUSTLL**: `v_i ≤ max_m_i` ∧ `v_j ≤ max_m_j`
- **MUSTUL**: `v_i ≥ min_m_i` ∧ `v_j ≤ max_m_j` (and the symmetric case)

If the joint system is infeasible but each single bound alone is feasible, the pair must
jointly change. This LP-feasibility form is the transparent equivalent of the published
bilevel search for moderate candidate pools; the bilevel machinery below is the efficient
search that backs the same guarantee.

## Bilevel → single-level (the keystone, `bilevel.py`)
Inner adversarial LP for objective `c` over fluxes `v`:

```
min cᵀv   s.t.  S v = 0 (λ free),  v ≥ lb (α ≥ 0),  v ≤ ub (β ≥ 0)
```

Dual:

```
max αᵀlb − βᵀub   s.t.  Sᵀλ + α − β = c,  α,β ≥ 0,  λ free
```

Reduction:
1. Write the inner LP in standard form (above).
2. Form its dual (above); `build_inner_dual` constructs λ/α/β and the feasibility rows.
3. Strong duality at optimality: `cᵀv == αᵀlb − βᵀub`.
4. Add primal feasibility + dual feasibility + the strong-duality equality to the outer
   model. Binary intervention variables gate bounds; the resulting bilinear `β·Δub·y`
   terms are linearised with big-M.
5. Every big-M comes from FVA bounds: `M = max(|min|,|max|) + buffer`
   (`bilevel.big_m_from_ranges`), never an arbitrary constant.

`bilevel.strong_duality_selftest` verifies steps 1–3 on a trivial LP in isolation
(roadmap step 3) before the reduction is relied upon.

## Stage 4 — FORCE set (`optforce.find_force_sets`)
Each MUST reaction carries a direction (MUSTU→"up", MUSTL→"down"; pair members inherit
their pair's direction). An intervention tightens the engineered strain's bounds:
- "up": `lower_bound := min_m` (force at least the target minimum)
- "down": `upper_bound := max_m` (cap at the target maximum)

A candidate set of ≤ k interventions is a valid FORCE set iff, on the engineered model
(WT bounds + interventions + a viability floor `biomass ≥ min_biomass_fraction · max_growth`),
the **worst-case** target flux still meets the threshold:

```
min  target_flux   s.t.  S v = 0,  engineered bounds,  biomass ≥ floor    ≥  threshold
```

Two solution paths (`find_force_sets(method=...)`):

- **`enumerate`** (default without Gurobi, LP-only): solve the worst-case LP for each
  `<=k` subset; smallest-first, skip supersets of valid sets. Exact and inspectable.
- **`milp`** (Gurobi, `bilevel.solve_force_milp`): one single-level MILP that *chooses*
  the interventions. The inner adversary LP is replaced by its dual; strong duality
  (`cᵀv = αᵀlb_eff(y) − βᵀub_eff(y)`) pins `v[target]` to the worst case, which the
  objective maximises over `y`. The bound gating `y[j]=1 ⇒ v[j] ≷ forced_j` and the
  bilinear dual terms `α·y`, `β·y` use **indicator constraints** (no big-M).
  Integer cuts enumerate alternative optima. Equation refs: Ranganathan et al. 2010,
  Eqs. 7–12.

Both must agree on the toy network (`tests/test_gurobi.py`). Note down-regulation alone
often fails to *force* production (the cell can drop uptake), which both paths reject.

## Validation targets
- Toy network with hand-computed MUST/FORCE sets (`tests/`, all passing).
- E. coli succinate case study (`examples/ecoli_succinate.py`) vs published results.
