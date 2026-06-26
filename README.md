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
              target_fraction=0.5, solver="gurobi")
of.compute_flux_ranges()
of.find_must_sets(max_order=2)
of.find_force_sets(k=3, n_solutions=10).to_dataframe()
```

See `CLAUDE.md` for the implementation plan and `docs/algorithm.md` for the math.
