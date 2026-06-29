# OptForce: algorithm and bilevel-to-single-level derivation

This is the **technical reference** for the four pipeline stages: the maths behind each
one, with pointers to the code that implements it. We re-derive the formulations from
Ranganathan et al. 2010 rather than translating the GAMS files.

For a slower, more conceptual walk-through aimed at *understanding why* each step is
shaped the way it is (what `target_fraction` means, why FVA yields a range, what the
second-order MUST sets buy us, how LP duality collapses the bilevel problem, and why the
MILP solutions are guaranteed to be valid steady states), read
[`in-depth.md`](in-depth.md) alongside this file.

## The idea in one paragraph

OptForce compares two pictures of the same network. The **wild type (WT)** is what the
cell does naturally. The **target ("M") strain** is the same network forced to
overproduce the product. For each reaction we ask: across *every* flux distribution the
network can adopt, how do its achievable flux values differ between the two pictures? If
a reaction's target flux is pushed entirely outside its WT range, that reaction *must*
change — it lands in a **MUST set**. The MUST sets are the raw material for the **FORCE
step**, which selects the smallest group of interventions (up-/down-regulation, knockout)
that *guarantees* overproduction no matter how the engineered cell redistributes its
flux.

Everything hinges on two ideas reused throughout: **flux ranges from FVA** (stages 1–2)
and an **adversarial "worst-case" inner problem** solved by LP duality (stages 3–4).

## Notation
- `S`: stoichiometric matrix (metabolites × reactions); `v`: flux vector; `lb`/`ub`:
  reaction bounds. A *steady state* is any `v` with `S v = 0` and `lb ≤ v ≤ ub`.
- WT model → ranges `[min_w, max_w]` per reaction (stage 1 FVA, `fva.wild_type_ranges`).
- Target model (constrained to ≥ target yield) → `[min_m, max_m]`
  (stage 2 FVA, `fva.target_ranges`).
- `c`: objective vector of the inner LP. For the FORCE step `c` selects the target
  reaction, so `cᵀv = v[target]`.

## Stages 1 & 2 — flux ranges

Both stages are ordinary **flux variability analysis (FVA)**: for each reaction, two LPs
maximise and minimise that reaction's flux over the feasible steady-state set. The only
difference between the stages is *which feasible set* — i.e. which constraints and which
"fraction of optimum" — they explore.

**Stage 1 — wild-type basal state (`fva.wild_type_ranges`).** The WT network at (a
fraction of) maximum growth. The `OptForce` driver runs this with
`fraction_of_optimum = wt_growth_fraction` (driver default `1.0`, i.e. growth pinned at
its maximum). This matters: the target model is the WT model *plus* an overproduction
floor, so the target feasible set is a **subset** of the unconstrained WT set. If WT FVA
also explored the whole unconstrained space, we would have `min_m ≥ min_w` and
`max_m ≤ max_w` for every reaction and **every MUST test would be vacuous**. Pinning the
WT at maximum growth makes the two range sets genuinely diverge — the WT picture becomes
"what a fast-growing cell does", which is what we want to contrast against forced
overproduction.

> Note on defaults: the standalone helper `fva.wild_type_ranges` defaults to
> `fraction_of_optimum = 0.0` (full WT space) because, called in isolation, the
> conservative reading is "any flux the WT *could* carry counts against declaring a
> forced change". The `OptForce` driver overrides this to `wt_growth_fraction = 1.0`
> for the reason above. Choose deliberately; the value changes which reactions look
> forced.

**Stage 2 — target ("M") model (`model.set_target_yield`).** Recipe: maximise
`target_reaction` to obtain its theoretical maximum `vmax`; then fix
`lower_bound := target_fraction · vmax` so *every* feasible distribution overproduces at
least that much. The biomass reaction stays the objective (the strain still grows).
Stage-2 FVA then uses `fraction_of_optimum = 0.0` because we want the full range of
fluxes *consistent with the target floor*, not just those tied to maximal biomass.

`target_fraction ∈ [0, 1]` is the design demand: "make the cell produce at least this
fraction of the theoretical maximum." Higher values force a sharper contrast (and more,
or larger, MUST sets); too high can make the strain non-viable. See `in-depth.md §1`.

## Stage 3 — MUST sets

### First order (`must_sets.first_order`) — interval logic
A single reaction is forced when its **entire** target range sits outside its WT range:
- **MUSTU** (must increase): `min_m > max_w` — even the lowest target flux exceeds the
  highest WT flux.
- **MUSTL** (must decrease): `max_m < min_w` — even the highest target flux is below the
  lowest WT flux.

Both are pure interval comparisons (a tolerance `tol` guards the boundary). A reaction
that is *not* first-order may still be free to take a WT-compatible value on its own — the
catch is whether it can do so *while its neighbours also satisfy the target*. That is what
second order tests.

### Second order (`must_sets.second_order`) — joint WT-feasibility
First order looks at one reaction at a time and so misses **coupled** requirements. A
pair `(i, j)` — neither already first order — is a second-order MUST pair when the
**wild-type** network cannot satisfy both target-side requirements *simultaneously*, even
though it can satisfy either one alone.

Operationally (`_wt_feasible_with`): impose single-sided bounds on the WT model and test
LP feasibility.
- **MUSTUU**: `v_i ≥ min_m_i` ∧ `v_j ≥ min_m_j`
- **MUSTLL**: `v_i ≤ max_m_i` ∧ `v_j ≤ max_m_j`
- **MUSTUL**: `v_i ≥ min_m_i` ∧ `v_j ≤ max_m_j` (and the symmetric `v_i ≤ max_m_i` ∧
  `v_j ≥ min_m_j`, stored as the swapped pair)

The classification rule is the same in all three cases:

> **joint system infeasible** *and* **each single bound feasible on its own** ⇒ the pair
> must jointly change.

The "each alone is feasible" clause is what excludes reactions already caught by first
order and ensures the *pair* — not one member — is the genuine unit of coupling. This
LP-feasibility form is the transparent equivalent of the published bilevel search for
moderate candidate pools; the bilevel machinery below is the efficient search that backs
the same guarantee for large pools. `max_pairs` caps the combinatorial cost; higher
orders (triples, …) generalise the rule but are gated off (`max_order ≤ 2`). See
`in-depth.md §2` for the rationale and a worked example.

## Bilevel → single-level (the keystone, `bilevel.py`)

OptForce's hard step is a **bilevel** problem: an *outer* decision (which interventions
to make) wraps an *inner* adversary (the cell picking the worst flux distribution it can,
subject to stoichiometry and the engineered bounds). We cannot nest a min inside a max and
hand it to a MILP solver directly. The fix is **strong duality of the inner LP**: replace
the inner minimisation by its dual constraints plus a strong-duality equality, collapsing
two levels into one.

**Inner primal LP** (the adversary), for a fixed objective `c` over fluxes `v`:

```
minimise   cᵀv
subject to S v = 0        (λ free)     -- mass balance
           v ≥ lb         (α ≥ 0)      -- lower bounds
           v ≤ ub         (β ≥ 0)      -- upper bounds
```

**Dual** (`build_inner_dual` constructs λ/α/β and the feasibility rows):

```
maximise   αᵀlb − βᵀub
subject to Sᵀλ + α − β = c
           α ≥ 0,  β ≥ 0,  λ free
```

**Reduction.**
1. Write the inner LP in the standard form above.
2. Form its dual (above).
3. **Strong duality** at optimality ties the two objectives together:
   `cᵀv == αᵀlb − βᵀub`. Because the primal is a feasible, bounded LP, the dual optimum
   *equals* the primal optimum — so adding the dual-feasibility constraints **plus** this
   equality forces `v` to the adversary's optimum without a nested solve.
4. Add primal feasibility + dual feasibility + the strong-duality equality to the outer
   model. Binary intervention variables `y` gate the bounds; an intervention shifts a
   bound by a constant, producing a bilinear term `β·(Δub)·y` (continuous dual × binary)
   — the *only* nonlinearity.
5. Linearise that product. Two interchangeable routes:
   - **indicator constraints** (Gurobi/CPLEX/SCIP): exact, no magic constant — used by
     `solve_force_milp`.
   - **big-M**: portable to any MILP backend, with `M` taken from FVA bounds via
     `bilevel.big_m_from_ranges` (`M = max(|min|, |max|) + buffer`), **never** an
     arbitrary constant.

`bilevel.strong_duality_selftest` verifies steps 1–3 on a trivial LP in isolation
(roadmap step 3) before the reduction is relied upon. A general primer on what the dual
*is* and why strong duality holds is in `in-depth.md §3`.

## Stage 4 — FORCE set (`optforce.find_force_sets`)

Each MUST reaction carries a direction (MUSTU→"up", MUSTL→"down"; pair members inherit
their pair's direction, `_candidate_directions`). Applying an intervention tightens the
**engineered** strain's bounds:
- "up":   `lower_bound := min_m` (force at least the target minimum)
- "down": `upper_bound := max_m` (cap at the target maximum)

A candidate set of ≤ `k` interventions is a valid **FORCE set** iff, on the engineered
model (WT bounds + interventions + a viability floor
`biomass ≥ min_biomass_fraction · max_growth`), the **worst-case** target flux still meets
the threshold:

```
min  v[target]   s.t.  S v = 0,  engineered bounds,  biomass ≥ floor    ≥  threshold
```

The `min` is the adversary again: we demand overproduction *even in the cell's least
cooperative steady state*. If the guaranteed worst case clears `threshold`, the set forces
the product. Two solution paths (`find_force_sets(method=...)`):

- **`enumerate`** (default without Gurobi, LP-only): solve the worst-case LP for each
  `≤ k` subset, smallest-first, skipping supersets of already-valid sets. Exact and fully
  inspectable.
- **`milp`** (Gurobi, `bilevel.solve_force_milp`): one single-level MILP that *chooses*
  the interventions. The inner adversary LP is replaced by its dual; strong duality
  (`cᵀv = αᵀlb_eff(y) − βᵀub_eff(y)`) pins `v[target]` to the worst case, which the
  objective maximises over `y`. Bound gating (`y[j]=1 ⇒ v[j] ≷ forced_j`) and the
  bilinear dual products (`α·y`, `β·y` via auxiliaries `p`, `q`) use **indicator
  constraints** (no big-M). Integer cuts enumerate alternative optima.
  Equation refs: Ranganathan et al. 2010, Eqs. 7–12.

Both paths must agree on the toy network (`tests/test_gurobi.py`). Note that
down-regulation **alone** often fails to *force* production (the cell can simply drop
uptake), and both paths correctly reject such sets.

**Why the solutions are genuine steady states.** Every flux vector the LP/MILP considers
satisfies `S v = 0` and the bounds, so it is a physically consistent steady-state
distribution — mass balance *is* the "everything chains up" guarantee, enforced at every
metabolite. The FORCE test is stronger than "a producing flux exists": by minimising the
target over the whole engineered polytope it certifies that *all* feasible distributions
overproduce. The constraint-based caveats (thermodynamics, kinetics, regulation) are
discussed in `in-depth.md §4`.

## Validation targets
- Toy network with hand-computed MUST/FORCE sets (`tests/`, all passing).
- E. coli succinate case study (`examples/ecoli_succinate.py`) vs published results.
