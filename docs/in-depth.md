# OptForce in depth — the *why* behind each stage

This document is the conceptual companion to [`algorithm.md`](algorithm.md). That file
is the terse reference (formulas + code pointers); this one slows down and explains the
reasoning, aimed at someone who knows the formulas but wants to understand *why they are
shaped the way they are*. It answers a specific set of questions:

1. Stages 1 & 2 — what `target_fraction` is for; whether the FVA here is "ordinary" FVA
   and how its constraints differ between the two stages; why FVA produces a **range** at
   all; and how that range is computed.
2. Stage 3 — what the **second-order** MUST sets are for: the problem they solve and how.
3. The **bilevel → single-level** reduction, preceded by a self-contained primer on
   **LP duality** in general.
4. Stage 4 — how the **MILP** relates to the dual, and the conceptual question of whether
   the minimal intervention sets are genuinely **feasible**: are the fluxes actually
   "chained up" through the network?

A running toy network (from `tests/conftest.py`) anchors the examples:

```
up_A: -> A        uptake, capacity 10
r1:   A -> B      branch to growth
r2:   A -> C      branch to product
bio:  B ->        biomass (cellular objective)
prod: C -> P
EX_P: P ->        product exchange (the target)
```

Steady state forces `up_A = r1 + r2`, `bio = r1`, `EX_P = prod = r2`. Carbon enters once
(≤ 10) and is split between a growth branch (`r1→bio`) and a product branch
(`r2→prod→EX_P`).

---

## 1. Stages 1 & 2 — flux ranges, `target_fraction`, and why FVA gives a range

### 1.1 What `target_fraction` is, and the rationale for it

OptForce never says "make the maximum possible amount of product". It says "make **at
least a fixed fraction** of the theoretical maximum". `target_fraction ∈ [0, 1]` is that
fraction, and `model.set_target_yield` turns it into a hard constraint in two steps:

```
vmax       = max v[target]        over the WT feasible space   (theoretical maximum yield)
threshold  = target_fraction · vmax
v[target] ≥ threshold             imposed as a lower bound on the target reaction
```

In the toy network the theoretical maximum of `EX_P` is 10 (all carbon to the product
branch), so `target_fraction = 0.5` sets `threshold = 5`: every flux distribution of the
target strain must export at least 5 units of product.

Why a *fraction* rather than the maximum, or an absolute number?

- **It is the design specification.** You are telling OptForce how ambitious the engineered
  strain must be. The whole procedure then answers "what is the minimal set of changes that
  *guarantees* at least this much product?" The fraction is the knob that defines "enough".
- **It normalises across products and models.** A fraction of the model's own theoretical
  maximum is comparable between targets and between models; an absolute mmol·gDW⁻¹·h⁻¹ value
  is not.
- **It controls the strength of the contrast — and the difficulty.** Push it toward 1.0 and
  you demand near-maximal production; the target feasible set shrinks toward the corner of
  the polytope where the cell does almost nothing *but* make product. That produces a
  sharper WT-vs-target contrast and generally more (or larger) MUST sets — but a strain
  forced to ≈100% of theoretical yield usually cannot also grow, so the engineered model
  can become non-viable or trivially over-constrained. Push it toward 0 and the demand is
  so weak the WT already satisfies it, MUST sets vanish, and OptForce finds nothing to do.
  Mid-range values (the published studies and the default `0.5`) keep the strain viable
  while still forcing a real change.

Crucially, `set_target_yield` constrains the *lower* bound of the target reaction and
leaves **biomass as the objective**. The target strain is still "a cell trying to grow",
not "a cell maximising product". The overproduction is imposed as a floor the cell cannot
sink below, which is exactly the engineering claim we want to end up proving: *no matter
how the cell behaves, it overproduces*.

### 1.2 Is this "ordinary" FVA? Yes — same procedure, different feasible set

Flux variability analysis is a standard constraint-based tool, and stages 1 and 2 use it
unchanged (`cobra.flux_analysis.flux_variability_analysis`, wrapped in `fva.py`). FVA asks,
for each reaction `j`: across the entire feasible steady-state space, what are the smallest
and largest values `v_j` can take? It returns `[min_j, max_j]` per reaction.

The two stages differ only in *which feasible set* they explore — i.e. the constraints and
the objective handed to FVA:

| | Stage 1 (WT, `wild_type_ranges`) | Stage 2 (target/"M", `target_ranges`) |
|---|---|---|
| Stoichiometry | `S v = 0` | `S v = 0` |
| Reaction bounds | native WT `lb`/`ub` | WT bounds **plus** `v[target] ≥ threshold = (target_fraction · vmax)` |
| Objective used as a floor | biomass, held at `fraction_of_optimum = wt_growth_fraction` (driver default **1.0**) | biomass, but FVA run at `fraction_of_optimum = 0.0` |
| Meaning | "what a maximally-growing wild type can do" | "everything consistent with the overproduction demand" |

Two design choices deserve explanation.

**Why the WT is pinned at maximum growth (`wt_growth_fraction = 1.0` in the driver).** The
target feasible set is the WT set with an *extra* constraint (`v[target] ≥ threshold`), so
it is a **subset** of the unconstrained WT set. If stage 1 also explored the full
unconstrained space, then for every reaction we would automatically have `min_m ≥ min_w`
and `max_m ≤ max_w` — the target range would sit *inside* the WT range and **no MUST test
could ever fire**. Pinning the WT at maximum growth changes the comparison from
"unconstrained vs constrained" to "a fast-growing cell vs a forced-overproducing cell",
which is a genuine contrast. (The standalone helper `fva.wild_type_ranges` defaults to
`0.0` for a different, conservative use; the OptForce driver overrides it to `1.0`. The
value you choose changes which reactions look forced, so choose deliberately.)

**Why stage 2 uses `fraction_of_optimum = 0.0`.** The overproduction floor is already
baked into the target model's bounds. We want the *full* span of fluxes compatible with
that floor — not only the ones coinciding with maximal biomass — so we do **not** further
pin biomass. Setting `fraction_of_optimum = 0` removes the biomass floor and lets FVA roam
the whole target-consistent polytope.

In the toy network this yields:

```
reaction   WT range [min_w,max_w]   target range [min_m,max_m]
up_A           [10, 10]                  [ 5, 10]
r1             [10, 10]                  [ 0,  5]
r2             [ 0,  0]                  [ 5, 10]
bio            [10, 10]                  [ 0,  5]
prod           [ 0,  0]                  [ 5, 10]
EX_P           [ 0,  0]                  [ 5, 10]
```

(WT pinned at growth = 10 forces all carbon down the growth branch, so the product branch
is exactly 0; the target floor `EX_P ≥ 5` forces ≥ 5 units down the product branch and
correspondingly caps growth.)

### 1.3 Why FVA gives a *range* and not a single number

A genome-scale (or even this toy) metabolic model is **underdetermined**: there are more
reactions than independent mass-balance equations. Formally, `S v = 0` defines a linear
subspace (the *null space* of `S`); intersecting it with the bound box `lb ≤ v ≤ ub` gives
a convex polytope — the *flux cone/polytope*. Unless that polytope is a single point, many
distinct flux vectors satisfy steady state and the bounds. They are all legitimate
"behaviours" of the same network.

Optimising an objective (e.g. biomass) does not remove this freedom:

- The objective may have **alternative optima** — different flux vectors achieving the same
  optimal growth.
- Reactions **not** coupled to the objective are free to vary even at fixed optimal growth.

FVA measures exactly this residual freedom, reaction by reaction. The *width* `max_j −
min_j` of a reaction's range is "how much room reaction `j` still has" given everything
else. A width of 0 means the reaction is pinned (fully determined); a wide range means it
is flexible.

In the toy network, at WT maximum growth `r1` is pinned to `[10, 10]` (all carbon must go
to growth to hit max biomass), while in the target strain it relaxes to `[0, 5]` (growth is
no longer maximal, so the growth branch has slack). That change of *range* — not of a
single flux value — is the raw signal OptForce reads.

### 1.4 How the ranges are computed

FVA is a batch of linear programs over one fixed feasible set. For `N` reactions:

1. (Optional objective floor.) If `fraction_of_optimum = f > 0`, first solve the FBA
   `Z* = max biomass`, then add the constraint `biomass ≥ f · Z*`. With `f = 1.0` biomass
   is pinned at its optimum; with `f = 0.0` no such constraint is added.
2. For each reaction `j`, solve **two** LPs over `{ S v = 0, lb ≤ v ≤ ub, (objective
   floor) }`:

   ```
   min_j = minimise v_j        max_j = maximise v_j
   ```

3. Collect `[min_j, max_j]`.

So a full FVA is `2N` LPs (cobra parallelises and warm-starts them). Each LP is solved by
the configured backend (`solvers.py`); the only thing that changes between stage 1 and
stage 2 is the constraint set described in §1.2. Nothing here is OptForce-specific — it is
textbook FVA, used as a measurement device.

---

## 2. Stage 3 — why we need the *second-order* MUST sets

### 2.1 What first order can and cannot see

First-order classification (`must_sets.first_order`) is a one-reaction-at-a-time interval
test: `MUSTU` if `min_m > max_w` (the target range lies entirely above the WT range),
`MUSTL` if `max_m < min_w` (entirely below). It catches every reaction whose flux, *on its
own*, is pushed completely outside its WT envelope. In the toy network that is `{r2, prod,
EX_P}` up and `{r1, bio}` down — and because the network is tiny, first order already
explains everything.

But "this reaction alone is forced out of its WT range" is a strict test. A reaction can be
perfectly capable of taking a WT-compatible value *by itself*, yet be **unable to do so at
the same time as its neighbours also satisfy the target**. First order, looking at one
reaction in isolation, is blind to that. The information lost is **coupling**.

### 2.2 The problem second order solves

The real question OptForce cares about is not "which single reactions are individually
forced?" but "which reactions must change to make overproduction unavoidable?" — and some
of those reactions only reveal themselves as a **group**. Second order answers:

> Which **pairs** of reactions must change *jointly*, because the wild-type network can
> satisfy either target requirement alone but not both together?

Why this matters for the downstream FORCE step: the MUST sets are the candidate pool of
interventions. If a genuinely necessary change is invisible to first order, it never enters
the candidate pool, and FORCE can miss the minimal intervention set entirely. Second order
widens the pool to include coupled reactions. (Third and higher orders generalise to
triples and beyond; they are combinatorially expensive and gated off behind `max_order`.)

### 2.3 How second order answers it

The mechanism (`must_sets.second_order`, via `_wt_feasible_with`) is a **joint feasibility
test on the wild-type model**. For a pair `(i, j)`, neither already first order, impose the
two target-side single-sided bounds *simultaneously* on the WT and ask the LP solver
whether the WT can still find a steady state. The classification rule, identical for all
three pair types:

```
joint system INFEASIBLE   and   each single bound FEASIBLE on its own
        ⇒   (i, j) must change jointly
```

with the three sign patterns

- `MUSTUU`: `v_i ≥ min_m_i` ∧ `v_j ≥ min_m_j`  (both forced up),
- `MUSTLL`: `v_i ≤ max_m_i` ∧ `v_j ≤ max_m_j`  (both forced down),
- `MUSTUL`: `v_i ≥ min_m_i` ∧ `v_j ≤ max_m_j`  (one up, one down; the symmetric case is
  stored as the swapped pair).

The "each single bound feasible alone" clause is essential. It guarantees neither reaction
is *individually* forced (those are already first order) — so the infeasibility is a
property of the **pair**, an emergent coupling, not of either member.

### 2.4 A worked example of coupling

The toy network is too small to have a second-order pair, so here is a minimal schematic
that does. Suppose the product needs a precursor `X`, and `X` can be supplied by two
redundant routes `a` and `b`, which share a budget in the wild type (e.g. a shared cofactor
or carbon pool):

```
WT coupling:   a + b ≤ 10
target needs:  a ≥ 8   and   b ≥ 8     (enough precursor through both routes)
```

Check the three conditions:

- `a ≥ 8` **alone**: feasible in WT (set `a = 8`, `b ≤ 2`). So `a` is **not** first-order
  `MUSTU`.
- `b ≥ 8` **alone**: feasible in WT (symmetric). So `b` is **not** first-order either.
- `a ≥ 8` **and** `b ≥ 8` together: requires `a + b ≥ 16`, but WT couples `a + b ≤ 10`.
  **Infeasible.**

Each alone is fine; together they break WT feasibility. The pair `(a, b)` is therefore
`MUSTUU` — a coupled requirement that first order, examining `a` and `b` separately, could
never detect. An engineering reading: relieving the shared budget (so both routes can run
hot at once) is the actual necessary intervention, and only the *joint* test surfaces it.
`MUSTUL` captures the analogous push-pull case — one reaction must rise while a competing
drain must fall, and the WT cannot do both at once.

> The LP-feasibility form above is the transparent equivalent of the published bilevel
> search for moderate candidate pools (`max_pairs` caps the pair count). For large pools
> the same guarantee is obtained more efficiently by the bilevel machinery of §3 — a
> single optimisation that hunts for the worst-case violating pair instead of enumerating
> all of them.

---

## 3. The dual problem, and the bilevel → single-level reduction

This is the conceptual keystone. We first build up **LP duality** in general, then show how
**strong duality** turns OptForce's two-level (bilevel) problem into one level.

### 3.1 Linear-programming duality from scratch

Every linear program (the **primal**) has a shadow twin (the **dual**) built from the same
data. Take the primal in the exact form OptForce's inner problem uses — minimise a linear
objective over fluxes subject to mass balance and bounds:

```
(P)   minimise   cᵀv
      subject to S v = 0          (assign dual variable λ, free)
                 v ≥ lb           (assign dual variable α ≥ 0)
                 v ≤ ub           (assign dual variable β ≥ 0)
```

Attach one dual variable to each constraint — a **price** on that constraint. Equality
constraints get a *free* (sign-unrestricted) price `λ`; inequality constraints get
*sign-restricted* prices (`α, β ≥ 0`). The dual is:

```
(D)   maximise   αᵀlb − βᵀub
      subject to Sᵀλ + α − β = c
                 α ≥ 0,  β ≥ 0,  λ free
```

**Where the dual comes from (one line of algebra).** Take any primal-feasible `v` and any
dual-feasible `(λ, α, β)`. Substitute `c = Sᵀλ + α − β`:

```
cᵀv = (Sᵀλ + α − β)ᵀ v = λᵀ(S v) + αᵀv − βᵀv = 0 + αᵀv − βᵀv
```

Now use the primal bounds with the sign of the dual prices: `v ≥ lb` with `α ≥ 0` gives
`αᵀv ≥ αᵀlb`; `v ≤ ub` with `β ≥ 0` gives `−βᵀv ≥ −βᵀub`. Hence

```
cᵀv  ≥  αᵀlb − βᵀub      for every primal-feasible v and dual-feasible (λ, α, β).
```

That inequality is **weak duality**: the dual objective is always a *lower bound* on the
primal objective (we are minimising the primal, so its twin maximises a quantity that can
never exceed it). It also explains, with no magic, *why* the dual objective is exactly
`αᵀlb − βᵀub` and *why* the prices carry those signs.

**Strong duality.** For a linear program that is feasible and bounded, the best lower bound
is not merely close but **exactly equal** to the primal optimum:

```
min cᵀv  =  max (αᵀlb − βᵀub).
```

The gap is zero. This is special to convex problems like LPs (in general there can be a
"duality gap"); for OptForce's inner LPs it always holds because they are feasible and
bounded by construction.

**Complementary slackness** (the fine print at the optimum): a price is nonzero only on a
binding constraint. `α_j > 0 ⇒ v_j = lb_j`, and `β_j > 0 ⇒ v_j = ub_j`. Economically the
duals are **shadow prices**: `α_j` is how much the optimal objective would improve per unit
of relaxation of lower bound `j` — the marginal value of loosening that constraint. (In
metabolic terms, the `λ` are sometimes read as metabolite "values" and `α, β` as the
worth of relaxing a reaction's capacity.)

`bilevel.strong_duality_selftest` checks all of this on a deliberately trivial LP before
the reduction is trusted:

```
min v1 + v2   s.t.  v1 − v2 = 0,  0 ≤ v1 ≤ 5,  1 ≤ v2 ≤ 4
```

Optimum `v1 = v2 = 1`, objective `2`. The dual `max (a2 − 5·b1 − 4·b2)` subject to
`lam + a1 − b1 = 1`, `−lam + a2 − b2 = 1` is maximised at `lam = 1, a1 = 0, a2 = 2, b1 =
b2 = 0`, objective `2`. Primal `=` dual `= 2`: strong duality, confirmed numerically in
isolation.

### 3.2 What "bilevel" means here, and why it is hard

OptForce's core question has two nested layers:

- **Outer (the engineer):** *choose* a set of interventions `y` (which reactions to up-/
  down-regulate or knock out), at most `k` of them.
- **Inner (the adversarial cell):** given those interventions, the cell *picks the flux
  distribution* — and for a worst-case guarantee we assume it picks the one **least**
  favourable to us, i.e. the steady state that **minimises** the product flux.

So the engineer is maximising (over `y`) something that is itself the result of a
minimisation (over `v`):

```
max_y   [  min_v  v[target]   subject to  S v = 0, engineered bounds(y), viability  ]
```

A `max` wrapped around a `min` is a **bilevel** program. You cannot hand it to a standard
MILP solver, because the inner `min` is not a set of constraints — it is another
optimisation. We need to flatten it.

### 3.3 The reduction: replace the inner min by its dual

Strong duality is the lever. The inner problem is an LP in `v`; by §3.1 its optimal value
equals its dual's optimal value, and — the key move — *its optimum is pinned by a set of
linear constraints*. Concretely, instead of "`v` solves the inner min", we assert three
linear conditions simultaneously:

1. **Primal feasibility:** `S v = 0`, engineered bounds.
2. **Dual feasibility:** `Sᵀλ + α − β = c`, `α, β ≥ 0`.
3. **Strong-duality equality:** `cᵀv = αᵀlb − βᵀub`.

Any `(v, λ, α, β)` satisfying 1–3 has `v` *equal to an inner optimum*: primal and dual
feasibility sandwich the objective by weak duality, and forcing equality (3) squeezes the
sandwich shut, so `v` cannot be anything but the worst-case flux. The nested minimisation
has become ordinary constraints. The outer problem is now a single-level model:

```
max_{y, v, λ, α, β}   v[target]
subject to            cardinality on y (≤ k),
                      conditions 1–3 with bounds gated by y.
```

One subtlety remains. An intervention shifts a bound by a constant *when its binary is on*
— e.g. an "up" intervention sets `lb_eff_j = lb_j + y_j·(forced_j − lb_j)`. Plugging this
into the dual objective `αᵀlb_eff` produces the term `α_j · y_j · (forced_j − lb_j)`: a
product of a **continuous** dual variable `α_j` and a **binary** `y_j`. That bilinear term
is the *only* nonlinearity left, and it is linearised exactly:

- **Indicator constraints** (`solve_force_milp`, Gurobi): introduce `p_j = α_j·y_j` with
  `y_j = 0 ⇒ p_j = 0` and `y_j = 1 ⇒ p_j = α_j`. No magic constant, no tolerance tuning.
- **Big-M** (portable fallback): `p_j ≤ M·y_j`, `p_j ≤ α_j`, `p_j ≥ α_j − M(1−y_j)`, with
  `M` taken from FVA bounds via `big_m_from_ranges` (`max(|min|,|max|) + buffer`), **never**
  an arbitrary number — an arbitrary `M` either cuts off valid solutions (too small) or
  wrecks numerics (too large).

That is the whole trick: **duality flattens the bilevel program, and a clean linearisation
removes the one bilinear term it leaves behind.**

---

## 4. Stage 4 — how the MILP relates to the dual, and whether the sets are *really* feasible

### 4.1 How the MILP relates to the dual problem

Stage 4 has two solution paths, and the dual is what distinguishes them.

- **`enumerate` (LP-only).** No dual at all. For each candidate subset of size ≤ `k`, it
  re-imposes the engineered bounds on the *actual* cobra model and solves the worst-case LP
  `min v[target]` directly (`_worst_case_target`). If the worst case clears the threshold,
  the subset is a valid FORCE set. This is the bilevel problem solved by brute force over
  the outer choices, with the inner min solved honestly as an LP each time. Transparent,
  exact, slow for large pools.

- **`milp` (single-level, dual-based).** This is the §3 reduction in action
  (`bilevel.solve_force_milp`). The inner `min` is *not* re-solved per subset; it is
  replaced once and for all by its dual feasibility constraints + the strong-duality
  equality. The binary `y` choosing interventions and the continuous `v, λ, α, β` live in
  **one** mixed-integer linear program. Strong duality pins `v[target]` to the inner
  worst case for *whatever* `y` the solver tries, so the outer `max` over `y` and the inner
  `min` over `v` are optimised **together** in a single solve. The solver chooses the
  interventions rather than us enumerating them; integer cuts then peel off alternative
  optimal FORCE sets.

So the relationship is: **the dual is precisely what makes the MILP possible.** Without it,
the inner minimisation cannot be expressed as constraints and the problem stays bilevel
(only the enumerate path can handle it). With it, the adversarial inner LP becomes linear
constraints, the "mixed-integer" part is just the intervention binaries plus the indicator
linearisation, and the whole bilevel collapses into one MILP. The two paths are required to
agree on the toy network (`tests/test_gurobi.py`), which cross-checks the dual reduction
against the brute-force truth.

### 4.2 Are the minimal subsets actually feasible? Do the fluxes "chain up"?

This is the deepest question, and the reassuring answer is that **network consistency is
not checked after the fact — it is built into the feasible set of every LP and MILP**.

**Mass balance *is* the chaining.** The constraint `S v = 0` is one equation per
metabolite: for each internal species, total rate of production = total rate of
consumption. There is no separate notion of "this reaction's output is routed to that
reaction"; instead, balance *at every node simultaneously* guarantees a globally consistent
flow. In the toy network, the row of `S` for metabolite `C` reads `r2 − prod = 0`, i.e.
whatever `r2` makes, `prod` must consume at the same rate — the fluxes cannot fail to line
up. You can never have flux appear from nowhere or vanish mid-pathway, because that would
violate some metabolite's balance row. So **any** `v` in the feasible set is, by
construction, a properly chained steady-state distribution. Every OptForce LP/MILP carries
`S v = 0` as a hard constraint, so every solution it returns is automatically network-
consistent. No post-hoc "do the fluxes connect?" check is needed; it is impossible for them
not to.

**Feasibility in the two senses you might mean.**

1. *Does the engineered strain even work (can it carry flux / grow)?* We add a viability
   floor `biomass ≥ min_biomass_fraction · max_growth`. If the interventions make the strain
   unable to meet that floor, the worst-case LP is **infeasible** and returns `None`
   (`_worst_case_target`), and the subset is rejected. A FORCE set that survives therefore
   admits at least the viable steady states we demanded.

2. *Is overproduction actually guaranteed?* This is stronger than "a producing flux
   exists." We take the **minimum** of the target over the *entire* engineered polytope. If
   even that worst case is `≥ threshold`, then **every** steady-state flux distribution the
   engineered cell could possibly adopt — subject to the interventions and viability — makes
   at least the threshold amount of product. There is no escape route left inside the
   feasible polytope. That is what "force" means, and it is why down-regulating the growth
   branch alone fails in the toy network: capping `r1` still lets the cell shrink `up_A`, so
   a zero-product steady state survives and the worst case is 0 < 5. The MILP path reaches
   the identical conclusion because strong duality makes its objective `v[target]` *equal*
   to that worst-case minimum over a genuine, fully mass-balanced polytope.

**What this does *not* guarantee (honest caveats).** The feasibility OptForce certifies is
**constraint-based / stoichiometric**: a mass-balanced steady state exists within the given
bounds. It does **not** by itself guarantee:

- *thermodynamic* feasibility (the solution may include thermodynamically infeasible
  internal loops unless loopless/`ΔG` constraints are added);
- correct *directionality* beyond whatever the reaction bounds encode;
- *kinetic* achievability or that enzyme levels can actually reach the forced fluxes;
- that cellular *regulation* will not counteract the intervention;
- that a real genetic edit realises exactly the assumed bound (`lb := min_m` or
  `ub := max_m`).

In other words, OptForce guarantees the intervention set is **sound at the steady-state
stoichiometric level** — the fluxes provably chain up and the worst case provably
overproduces — which is the right and standard guarantee for this class of method, but it is
a *model-level* certificate, not an in-vivo one. That is exactly why the roadmap ends with
validation against the published *E. coli* succinate case study
(`examples/ecoli_succinate.py`): the maths is checked in isolation (`strong_duality_selftest`,
the toy tests), and the biology is checked against a known real-world result.

---

## Where to look in the code

| Concept | Module / function |
|---|---|
| `target_fraction` → overproduction floor | `model.set_target_yield`, `model.theoretical_max` |
| Stage 1 / 2 FVA | `fva.wild_type_ranges`, `fva.target_ranges`, `fva.compute_flux_ranges` |
| First-order MUST | `must_sets.first_order` |
| Second-order MUST (joint feasibility) | `must_sets.second_order`, `must_sets._wt_feasible_with` |
| Inner dual construction | `bilevel.build_inner_dual` |
| Strong-duality isolation test | `bilevel.strong_duality_selftest` |
| FORCE worst-case LP (enumerate) | `optforce.OptForce._worst_case_target`, `find_force_sets` |
| FORCE single-level MILP (dual) | `bilevel.solve_force_milp` |

See [`algorithm.md`](algorithm.md) for the matching formula-level reference.
