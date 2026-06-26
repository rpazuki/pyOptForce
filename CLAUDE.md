# CLAUDE.md — pyOptForce

Guidance for Claude Code when working in this repository.

---

## 1. Project goal

`pyOptForce` is a clean-room Python implementation of the **OptForce** procedure
(Ranganathan, Suthers & Maranas, *PLoS Comput Biol* 2010), ported from the MATLAB
COBRA Toolbox so that the full simulation pipeline is under our control and can be
**extended** — new MUST-set classification rules, alternative inner objectives,
custom constraints, regularisation, and integration with downstream analysis.

We are **not** wrapping the MATLAB/GAMS code. We re-implement the algorithm on top of
the Python constraint-based modelling stack so every optimisation problem is visible
and modifiable.

### Why build rather than reuse
- COBRApy, cameo, and StrainDesign do **not** ship OptForce.
- We need to extend the core algorithm (new MUST tiers, alternative objectives,
  measured-flux integration), which requires owning the MILP formulations.

---

## 2. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Model I/O & data structures | **cobrapy** (`cobra`) | SBML loading, reactions, metabolites, FBA, FVA |
| Optimisation interface | **optlang** (via cobra) + direct **gurobipy** / **cplex** / **pyscipopt** for bilevel MILPs | optlang is fine for LPs and simple MILPs; bilevel duality linearisation is clearer with a direct solver API |
| Numerics | numpy, pandas | flux vectors, MUST-set tables |
| Testing | pytest | small toy network with known OptForce sets |
| Packaging | pyproject.toml (PEP 621) | editable install via `pip install -e .` |
| Python | 3.11+ | |

**Solver priority:** prefer Gurobi or CPLEX (academic licences) for the MILP steps;
fall back to SCIP (`pyscipopt`, open source) and GLPK for LP-only steps. Indicator
constraints (Gurobi/CPLEX/SCIP) avoid the numerical issues of big-M formulations.

---

## 3. The OptForce algorithm — implementation reference

OptForce contrasts the flux ranges of a **wild-type (WT)** strain with those of a
target **overproducing** strain and identifies the minimal set of interventions
(up-regulation, down-regulation, knockout) that *force* the desired overproduction.

Pipeline stages (each is a separate module — see §4):

1. **WT flux ranges** — FVA on the wild-type model.
   → `minFluxesW`, `maxFluxesW` for every reaction.

2. **Target/overproducing flux ranges** — FVA on the model constrained to the target
   product yield (e.g. ≥ some fraction of theoretical max).
   → `minFluxesM`, `maxFluxesM`.

3. **MUST-set classification** — compare the two range sets to find reactions whose
   flux *must* change to achieve the target:
   - `MUSTU`: flux must **increase** (first order)
   - `MUSTL`: flux must **decrease** (first order)
   - `MUSTUU`, `MUSTLL`, `MUSTUL`: second-order sets — pairs of reactions that must
     *jointly* change, found via bilevel MILP. Higher orders (triples, …) generalise
     the same rule but are combinatorially expensive; gate behind a `max_order` param.

   First order is a simple interval comparison. Second+ order requires solving a
   **bilevel MILP**: maximise the "violation" of the WT-consistent flux while the
   inner problem enforces stoichiometric feasibility. Linearise the inner problem via
   its dual / KKT conditions (or solver indicator constraints).

4. **OptForce MILP (FORCE step)** — given the MUST sets, find the minimal set of `k`
   interventions (the "FORCE set") guaranteeing the product is overproduced at the
   target level for *all* feasible flux distributions of the mutant. Enumerate
   alternative optimal FORCE sets (integer cuts / solution pool).

### Key formulation notes (the hard part)
- The bilevel structure (outer: choose interventions; inner: adversarial flux
  distribution) is reduced to a single-level MILP using strong duality of the inner LP.
- Big-M constants must be chosen from FVA bounds, not arbitrary — document each M.
- Reversible reactions: split or handle with signed bounds consistently across stages.
- Validate against the published *E. coli* succinate case study and the COBRA Toolbox
  tutorial outputs.

---

## 4. Repository layout

```
pyOptForce/
├── CLAUDE.md                  # this file
├── README.md
├── pyproject.toml
├── src/pyoptforce/
│   ├── __init__.py            # public API exports
│   ├── model.py               # load/prepare models, target setup, reversibility handling
│   ├── fva.py                 # stage 1 & 2: WT and target FVA wrappers
│   ├── must_sets.py           # stage 3: first- and higher-order MUST classification
│   ├── bilevel.py             # bilevel MILP builder + dual linearisation (shared engine)
│   ├── optforce.py            # stage 4: FORCE-set MILP, solution enumeration
│   ├── solvers.py             # solver selection, big-M / indicator helpers, config
│   ├── results.py             # result containers (dataclasses), pandas exporters
│   └── extensions/            # <-- your extension point; keep core stages pure
│       └── __init__.py
├── tests/
│   ├── test_fva.py
│   ├── test_must_sets.py
│   ├── test_optforce_toy.py   # toy network with hand-verified sets
│   └── data/
├── examples/
│   └── ecoli_succinate.py     # reproduce the canonical case study
└── docs/
    └── algorithm.md           # detailed math: bilevel → single-level derivation
```

---

## 5. Public API (target design)

```python
import cobra
from pyoptforce import OptForce

model = cobra.io.read_sbml_model("iJO1366.xml")

of = OptForce(
    model,
    target_reaction="EX_succ_e",   # product exchange
    biomass_reaction="BIOMASS_Ec_iJO1366_core_53p95M",
    target_fraction=0.5,           # >= 50% of theoretical max yield
    solver="gurobi",
)

of.compute_flux_ranges()          # stages 1 & 2
must = of.find_must_sets(max_order=2)   # stage 3
force_sets = of.find_force_sets(k=3, n_solutions=10)  # stage 4

force_sets.to_dataframe()
```

Keep each stage independently callable and inspectable — intermediate results
(`minFluxesW`, MUST tables, MILP objects) must be accessible attributes, not hidden
locals. This is what makes the algorithm extensible.

---

## 6. Conventions

- **Type hints everywhere**; `dataclasses` for result objects.
- **No silent solver failures** — raise on non-optimal status (mirror cameo's policy).
- Keep the **core stages solver-agnostic** where possible; isolate solver-specific code
  in `solvers.py` and `bilevel.py`.
- Every MILP gets a docstring stating its decision variables, constraints, objective,
  and the source equation numbers in Ranganathan et al. 2010.
- Document every big-M with where its value comes from.
- Tests must include at least one network where the FORCE set is known by hand.

---

## 7. Implementation roadmap (suggested order for Claude Code)

1. `model.py` + `fva.py` — load a model, run WT FVA, set up the target-constrained
   model, run target FVA. Verify against `cobra.flux_analysis.flux_variability_analysis`.
2. `must_sets.py` first order (`MUSTU`, `MUSTL`) — pure interval logic. Test on toy net.
3. `bilevel.py` — the dual-linearisation engine. This is the keystone; get it right and
   unit-test the LP duality on a trivial problem before wiring it in.
4. `must_sets.py` second order using `bilevel.py`.
5. `optforce.py` — FORCE-set MILP + integer-cut enumeration.
6. `examples/ecoli_succinate.py` — end-to-end validation against published results.
7. Only then build out `extensions/`.

**Do not** jump to stage 4 before the bilevel engine in step 3 is tested in isolation.

---

## 8. References

- Ranganathan S, Suthers PF, Maranas CD (2010). OptForce: An Optimization Procedure for
  Identifying All Genetic Manipulations Leading to Targeted Overproductions.
  *PLoS Comput Biol* 6(4): e1000744.
- COBRA Toolbox OptForce module (MATLAB) — reference for stage decomposition and
  function signatures (`FVAOptForce`, `findMustU/L`, `findMustUU/LL/UL`, `optForce`).
- COBRApy docs: https://opencobra.github.io/cobrapy/
- optlang docs: https://optlang.readthedocs.io/

> Note: the COBRA Toolbox OptForce is based on GAMS files from the Maranas group. We
> re-derive the formulations from the paper rather than translating GAMS line-by-line.
