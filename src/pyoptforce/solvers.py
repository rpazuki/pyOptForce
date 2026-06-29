"""Solver selection and backend-specific helpers.

Keep all solver-specific code here. The MUST/FORCE MILPs are built on top of
**optlang** (the same optimisation layer cobra uses) so the rest of the code stays
solver-agnostic: we only ever touch ``optlang`` Model/Variable/Constraint objects.

Solver priority follows CLAUDE.md: prefer Gurobi/CPLEX, then SCIP, then GLPK. We
discover what is actually installed via ``optlang.available_solvers`` and map the
friendly name to the optlang interface module.

Indicator constraints (Gurobi/CPLEX/SCIP) would avoid the numerical pitfalls of
big-M, but they are not portable across every backend. To keep one code path that
works everywhere (including GLPK), the bilevel reduction uses **big-M** gating with
constants derived from FVA bounds (see ``bilevel.big_m_from_ranges``); each big-M is
documented at its use site.
"""

from __future__ import annotations

import optlang

# friendly name -> optlang interface attribute / available_solvers key
_NAME_TO_KEY = {
    "gurobi": "GUROBI",
    "cplex": "CPLEX",
    "scip": "SCIP",
    "glpk": "GLPK",
}

# preference order when the caller asks for "auto" or an unavailable solver
_PREFERENCE = ("gurobi", "cplex", "scip", "glpk")

SUPPORTED = ("gurobi", "cplex", "scip", "glpk")


def available() -> list[str]:
    """Friendly names of MILP-capable backends that are actually importable."""
    avail = optlang.available_solvers
    return [name for name, key in _NAME_TO_KEY.items() if avail.get(key, False)]


def get_interface(name: str = "auto"):
    """Return the optlang interface module for ``name``.

    ``name="auto"`` (or an unavailable solver) falls back to the best installed
    backend by :data:`_PREFERENCE`. Raises if nothing usable is installed.
    """
    avail = available()
    if not avail:
        raise RuntimeError(
            "No MILP-capable optlang backend installed. "
            "Install one of: gurobi, cplex, scip (pyscipopt), glpk."
        )

    chosen: str | None = None
    if name not in ("auto", None):
        if name not in _NAME_TO_KEY:
            raise ValueError(f"Unknown solver {name!r}; supported: {SUPPORTED}")
        if name in avail:
            chosen = name
    if chosen is None:
        chosen = next(n for n in _PREFERENCE if n in avail)

    key = _NAME_TO_KEY[chosen]
    return getattr(optlang, f"{key.lower()}_interface")


def cobra_solver_name(name: str = "auto") -> str:
    """Map a friendly solver name to the string cobra expects in ``model.solver``."""
    avail = available()
    if name in avail:
        return name
    return next(n for n in _PREFERENCE if n in avail)


def get_backend(name: str):
    """Backwards-compatible alias for :func:`get_interface`."""
    return get_interface(name)


def gurobi_available() -> bool:
    """True if ``gurobipy`` can be imported (academic licence installed)."""
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    return True


def require_gurobi():
    """Import and return the ``gurobipy`` module, or raise a clear error.

    The direct-gurobipy MILP path (bilevel FORCE reduction) needs the real solver
    API for indicator constraints and the solution pool; optlang is not enough.
    """
    try:
        import gurobipy
    except Exception as exc:  # pragma: no cover - exercised only without gurobi
        raise RuntimeError(
            "gurobipy is not installed. Install Gurobi (academic licence) and "
            "`pip install gurobipy` to use the MILP FORCE path; the LP path works "
            "on any backend."
        ) from exc
    return gurobipy
