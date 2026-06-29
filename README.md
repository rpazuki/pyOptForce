# pyOptForce

A clean-room, extensible Python implementation of the **OptForce** strain-design
procedure (Ranganathan, Suthers & Maranas, *PLoS Comput Biol* 2010), built on
COBRApy + optlang. Ported from the MATLAB COBRA Toolbox so the full simulation
pipeline is under your control and open to extension.

## Why
COBRApy, cameo, and StrainDesign do not ship OptForce. This project re-implements the
algorithm from the paper so every optimisation problem is visible and modifiable.

## Install (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,scip]"     # add "gurobi" if you have a Gurobi licence
```

## Quick start
```python
import cobra
from pyoptforce import OptForce

model = cobra.io.read_sbml_model("iJO1366.xml")
of = OptForce(model, target_reaction="EX_succ_e",
              biomass_reaction="BIOMASS_Ec_iJO1366_core_53p95M",
              target_fraction=0.5, solver="auto")  # auto-discovers an installed backend
of.compute_flux_ranges()                # stages 1 & 2 (WT + target FVA)
of.find_must_sets(max_order=2)          # stage 3 (interval + joint-feasibility)
of.find_force_sets(k=3, n_solutions=10).to_dataframe()   # stage 4 (FORCE sets)
```

Every intermediate result stays on the instance (`of.flux_ranges`, `of.must_sets`,
`of.force_sets`). `solver="auto"` picks Gurobi → CPLEX → SCIP → GLPK, whichever is
installed; the LP-only path runs on GLPK alone.

Runnable example: `python examples/ecoli_succinate.py` (e_coli_core succinate).

See `CLAUDE.md` for the implementation plan and `docs/algorithm.md` for the math.
