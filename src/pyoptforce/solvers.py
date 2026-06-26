"""Solver selection and backend-specific helpers.

Keep all solver-specific code here. Prefer indicator constraints (Gurobi/CPLEX/SCIP)
over big-M where available. LP-only steps can use optlang/GLPK.
"""

from __future__ import annotations

SUPPORTED = ("gurobi", "cplex", "scip", "glpk")


def get_backend(name: str):
    """Return a thin adapter exposing the MILP primitives we need
    (add var, add constraint, add indicator constraint, optimise, solution pool)."""
    raise NotImplementedError
