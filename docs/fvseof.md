# FVSEOF — finding and reverse-checking over-expression targets

`pyoptforce.extensions.fseof` implements **FVSEOF** (Flux Variability Scanning based on
Enforced Objective Flux) as an extension alongside the core OptForce pipeline, plus a
strengthening of its target-selection rule that borrows OptForce's *no-overlap* logic.

This document is in three parts, in the order you should read them:

1. **What FVSEOF is** and the two ways we use it.
2. **How it compares to OptForce** — specifically why OptForce's "no-overlap ⇒ *must*"
   test is stronger than FVSEOF's monotonic-increase test.
3. **Our update**: replacing the weak midpoint-slope criterion with a *min-increase* tier
   and an OptForce-style *necessity* tier.

Throughout, the frame is a concrete lab question: **a gene was over-expressed and the
growth rate changed — can the model find that target ahead of time (forced increase), and
can it confirm the observed target after the fact (reverse-check)?**

---

## 1. What FVSEOF is

FVSEOF is a cheap (LP/FVA-only) screen for **gene amplification / over-expression
targets** — reactions worth pushing *up* to improve a phenotype. It was introduced for
product over-production (Choi et al. 2010 for the single-flux form FSEOF; Park et al. 2012
for the flux-variability form FVSEOF), and it works by a simple intuition:

> An enzyme worth over-expressing is one whose flux **naturally rises** as you demand more
> of the objective.

So you *enforce* a rising sequence of an objective flux and watch which internal fluxes
get dragged up with it.

### The scan

Pick an **enforced reaction** — the flux you will force upward. Sweep it over `n_steps`
levels from `lo` to `hi`:

- `hi` defaults to the reaction's **theoretical maximum** (`model.theoretical_max`);
- `lo` defaults to its **wild-type flux at maximum growth**.

At each level the enforced flux is *pinned* (`bounds = (level, level)`) and **FVA** is run
over every other reaction, giving a range `[v_min(level), v_max(level)]`. Writing **WT**
for the bottom rung (`levels[0]`, least enforcement) and **top** for the top rung
(`levels[-1]`, maximal enforcement), the published FVSEOF criterion ranks a reaction as an
amplification target when the **midpoint** of its range,

```
mid(level) = ½ · (v_min(level) + v_max(level)),
```

**trends upward** with the enforced level (positive least-squares slope). Down-regulation
(attenuation) targets are the mirror image (negative slope).

### The two use cases in this codebase

Both are the same scan pointed at a different enforced reaction:

| Use case | `enforced_reaction` | `lo` | `hi` | Question answered |
|---|---|---|---|---|
| **Forced increase** (design) | a product exchange, or biomass | WT baseline (default) | theoretical max (default) | *Which reactions should I over-express to push this up?* |
| **Reverse-check** (validation) | biomass (growth) | **observed growth rate** | model max growth (default) | *Is the reaction I over-expressed in the lab a target the model supports, and how strongly?* |

The result keeps a row for **every** scanned reaction, so reverse-checking a specific
gene's reaction is a lookup:

```python
from pyoptforce.extensions.fseof import fvseof

# Forced-increase screen for succinate over-production
res = fvseof(model, "EX_succ_e", biomass_reaction="BIOMASS_Ecoli_core_w_GAM", n_steps=8)
res.amplification_targets()          # ranked up-regulation targets, strongest first
res.attenuation_targets()            # ranked down-regulation targets

# Reverse-check: given an observed growth rate, is PGI a supported target?
mu_obs = 0.3 * model.slim_optimize()             # a measured, sub-maximal growth rate
res = fvseof(model, "BIOMASS_Ecoli_core_w_GAM", lo=mu_obs, n_steps=8)
res.check("PGI")                     # the FvseofRow: verdict + all flags
print(res.interpret("PGI"))          # one-line, human-readable verdict for the lab
```

---

## 2. How it compares to OptForce

OptForce's MUST-set stage asks a **necessity** question with a very sharp test. For a
first-order up set (MUSTU) it compares the wild-type flux range `[min_W, max_W]` against
the target-constrained range `[min_M, max_M]` and flags the reaction when the intervals
are **disjoint above**:

```
MUSTU  ⇔  min_M > max_W
```

If the target-strain range does not overlap the WT range, then **every** feasible
target-achieving flux distribution runs that reaction higher than **any** WT distribution
could — the reaction's increase is *forced*. That is a logical guarantee over the whole
feasible space, not a trend.

FVSEOF's midpoint-slope criterion is a much weaker, **associative** statement. The two
methods sit in different logical categories:

| | FVSEOF (midpoint slope) | OptForce (MUST) |
|---|---|---|
| Machinery | FVA only (LP) | bilevel → single-level MILP for 2nd order; interval test for 1st |
| Claim | *association* — flux tends to rise with the objective | *necessity* — flux must change or the target is unreachable |
| Evidence | slope of a chosen representative of the range | disjointness of two ranges |
| Cost | seconds | LP (1st order) to MILP (higher order) |
| Failure mode | flags reactions that need not actually rise | — |

### Why "monotonic increase" is weak

The midpoint is only one point inside `[v_min, v_max]`, and at any enforced level the cell
is free to sit **anywhere** in that interval. A rising midpoint therefore says only that
the feasible *envelope* drifts upward — **not** that any individual flux is obliged to
rise. Concretely:

> WT range of reaction `R`: `[0, 10]`. Under enforced production, `R`'s range: `[2, 12]`.
> The midpoint rose 5 → 7, so plain FVSEOF flags `R`. **But the ranges overlap on
> `[2, 10]`**: there is a perfectly feasible high-production state with `R = 2`, *below*
> WT's midpoint. Over-expressing `R` is neither required nor guaranteed to help.
> OptForce correctly refuses `R` as MUSTU because `min_M = 2 < max_W = 10`.

The variability-narrowing heuristic sometimes bolted onto FVSEOF doesn't fix this: a range
can narrow around a value that still sits inside the WT range. Narrowing ≠ disjoint.

---

## 3. Our update: min-increase and necessity tiers

We keep FVSEOF's cheap scan but throw away its weak decision rule. Instead we read **three
tiers of increasing logical strength** off the same swept profiles, so the user sees not
just *whether* a reaction is a candidate but *how trustworthy* the signal is.

For each reaction, with WT = bottom rung and top = top rung:

| Tier | Flag | Test | Meaning |
|---|---|---|---|
| 1 — weak (classic) | `slope > 0` | midpoint least-squares slope positive | the envelope drifts up (associative) |
| 2 — **min-increase** (our first strengthening) | `min_monotone_up` | `v_min` non-decreasing across the sweep **and** `v_min(top) > v_min(WT)` | the **worst case** rises — a stronger constraint than a midpoint trend |
| 3 — **necessity** (OptForce logic) | `must_up` | `v_min(top) > v_max(WT)` | the top-rung range is **disjoint above** the WT range — the increase is *forced* |

Tier 3 is precisely OptForce's `min_M > max_W`, evaluated across the swept envelope (the
top rung being the maximal-enforcement analogue of the target-constrained model). It is a
**graded MUSTU**: a genuine necessity guarantee recovered from an LP-only scan.

Down-regulation mirrors this exactly: `max_monotone_down` (worst-case high end falls) and
`must_down` (`v_max(top) < v_min(WT)`, range disjoint *below* WT).

The scan collapses these into a single `verdict` per reaction, strongest applicable label
winning:

```
must_up  >  amplification (min-increase + positive slope)  >  weak_up  >  none
must_down > attenuation   (min-decrease + negative slope)  >  weak_down
```

### These tiers are a strength ordering, not nested sets

`must_up` deliberately **wins over** every weaker label, even when the classic slope
criterion would *miss* the reaction. A range can shift bodily upward (so `v_min(top)`
clears `v_max(WT)`) while its `v_max` collapses, dragging the **midpoint slope negative**.
Such a reaction is a genuine necessity that plain FVSEOF would discard — a *false negative*
of the weak method. That mismatch is not a bug in the tiering; it is the whole reason the
necessity tier exists. So do not expect `{must_up} ⊆ {slope > 0}` in general; expect the
necessity tier to *catch reactions the slope misses* and to *demote reactions the slope
over-sells*.

### Worked toy example

On the toy network (`tests/conftest.py`): shared uptake `up_A` (cap 10) feeds a growth
branch (`r1 → bio`) and a product branch (`r2 → prod → EX_P`), with
`up_A = r1 + r2`, `bio = r1`, `EX_P = prod = r2`.

**Forced increase, enforcing the product `EX_P` from 0 → 10:**

| reaction | verdict | slope | `min_monotone_up` | `must_up` | WT range | top range |
|---|---|---|---|---|---|---|
| `r2`   | **must_up** | +1.0 | ✓ | ✓ | `[0, 0]` | `[10, 10]` |
| `prod` | **must_up** | +1.0 | ✓ | ✓ | `[0, 0]` | `[10, 10]` |
| `up_A` | amplification | +0.5 | ✓ | ✗ | `[0, 10]` | `[10, 10]` |
| `r1`   | attenuation | −0.5 | — | — | `[0, 10]` | `[0, 0]` |
| `bio`  | attenuation | −0.5 | — | — | `[0, 10]` | `[0, 0]` |

The instructive row is **`up_A`**. Its worst case *does* rise monotonically
(`min_monotone_up = ✓`) and its midpoint slope is positive — so both weak criteria flag
it. Yet it is **not necessary** (`must_up = ✗`): the wild type can already run `up_A` at 10
(by sending all carbon to growth), so its WT range `[0, 10]` is *not* disjoint from the
top range `[10, 10]`. Only `r2` and `prod` are forced. And indeed
`necessary_up() ∪ {EX_P} = {r2, prod, EX_P}` reproduces OptForce's MUSTU on this network
exactly — a cross-check enforced in `tests/test_fvseof.py`.

**Reverse-check, enforcing biomass from an observed 4.0 → max 10:** `r1` becomes
`must_up` (you cannot grow faster without pushing the growth branch), while the product
branch (`r2`, `prod`, `EX_P`) turns to `attenuation` (it must be surrendered to grow). If
the gene you over-expressed in the lab maps to `r1`, the model *necessarily* supports it;
if it maps to `up_A`, the support is real but weaker (favoured, not forced); if it maps to
something with `verdict = "none"`, the model does not couple it to growth at all.

---

## 4. The capacity caveat (read before using growth as the target)

FVSEOF, like OptForce, sees only **stoichiometry and flux bounds** — never enzyme
abundance. Over-expressing a reaction that is not pressed against a binding upper bound is
a **no-op** in the model. This bites hardest when the enforced flux is **growth itself**:

- In a plain FBA model the wild type is **already at its growth optimum**. There is no
  flux redistribution that yields more growth, so enforcing growth *above* the optimum is
  infeasible and the scan is empty.
- Therefore the reverse-check requires `lo < hi` with **real headroom**. That headroom is
  physical only if either (a) the **observed** growth rate genuinely lies *below* the
  model's FBA maximum (common — cells rarely hit the FBA bound), or (b) you use an
  **enzyme-constrained model** (GECKO / sMOMENT / MOMENT: `flux ≤ enzyme · k_cat`) in
  which over-expression *relaxes a binding capacity*.

The implementation makes this explicit: if `lo` defaults to the WT max-growth flux and
that already equals `hi` (the theoretical max), `fvseof` raises a `ValueError` about
"headroom" rather than silently returning nothing. A lab over-expression that lifts growth
*above* the current FBA optimum simply cannot be represented by a purely stoichiometric
model — that phenotype lives in the enzyme-capacity layer the model omits.

---

## 5. API reference

```python
fvseof(
    model,
    enforced_reaction,             # product exchange OR biomass reaction
    *,
    biomass_reaction=None,         # anchors the WT baseline to max growth; optional floor
    lo=None,                       # bottom rung; default = WT max-growth flux of enforced
    hi=None,                       # top rung;    default = theoretical max of enforced
    n_steps=10,                    # number of enforced levels (>= 2)
    reactions=None,                # restrict the scan (and FVA cost) to these ids
    biomass_floor_fraction=0.0,    # keep biomass >= fraction*max at every rung (viability)
    tol=1e-6,
) -> FvseofResult
```

`FvseofResult`:

| Member | Returns |
|---|---|
| `to_dataframe()` | one row per scanned reaction, strongest verdict first |
| `amplification_targets(min_verdict=None)` | up-regulation targets; `min_verdict` sets the weakest tier kept (`must_up` > `amplification` > `weak_up`) |
| `attenuation_targets(min_verdict=None)` | down-regulation targets (mirror) |
| `necessary_up()` | reactions with the `must_up` necessity guarantee |
| `check(reaction_id)` | the `FvseofRow` for one reaction (reverse-check); `KeyError` if unscanned |
| `interpret(reaction_id)` | one-line, lab-facing verdict string |
| `levels`, `profiles` | the raw enforced levels and per-reaction `[v_min, v_max]` sweep, for plotting/inspection |

`FvseofRow` fields: `reaction`, `verdict`, `slope`, `must_up`, `must_down`,
`min_monotone_up`, `max_monotone_down`, `vmin_wt`, `vmax_wt`, `vmin_top`, `vmax_top`, plus
`is_amplification` / `is_attenuation` convenience properties.

> **Signed-flux convention.** The scan works in signed-flux space (consistent with the
> rest of pyOptForce — reversible reactions are kept signed, not split). "Increase" means
> *more positive*. A reaction carrying net-negative flux whose magnitude grows (e.g. a
> substrate uptake as growth is forced up) therefore appears under the *down* verdicts;
> interpret those as magnitude increases. The raw `profiles` are available if you need to
> post-process by `|flux|`.

---

## 6. Tests

`tests/test_fvseof.py`:

- **Toy network** (every verdict hand-derivable): necessity tier reproduces OptForce's
  MUSTU (`necessary_up() ∪ {enforced} == MUSTU`); the `up_A` case pins down that
  min-increase is strictly stronger than the midpoint slope yet strictly weaker than
  necessity; hand-exact verdicts for all reactions; the reverse-check-via-observed-growth
  path; the no-headroom capacity `ValueError`; filter/lookup/validation behaviour.
- **`e_coli_core`** (real model): forced-succinate scan is internally self-consistent
  (every verdict reproducible from its flags, every necessity flag backed by genuine range
  disjointness); the default-baseline growth scan raises the capacity error; and the
  observed-rate reverse-check runs end-to-end with a per-reaction `interpret`.

## References

- Choi HS, Lee SY, Kim TY, Woo HM (2010). *In silico identification of gene amplification
  targets for improvement of lycopene production.* Appl. Environ. Microbiol. 76(10):3097–3105.
- Park JM, Park HM, Kim WJ, Kim HU, Kim TY, Lee SY (2012). *Flux variability scanning based
  on enforced objective flux for identifying gene amplification targets.* BMC Syst. Biol. 6:106.
- Ranganathan S, Suthers PF, Maranas CD (2010). *OptForce…* PLoS Comput. Biol. 6(4):e1000744.
  (the MUST/no-overlap logic this extension borrows — see `docs/algorithm.md`).
