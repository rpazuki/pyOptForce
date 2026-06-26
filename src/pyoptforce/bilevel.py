"""Bilevel MILP engine — the keystone of the higher-order MUST sets and FORCE step.

OptForce's outer problem chooses interventions; the inner problem is an adversarial
flux distribution constrained by stoichiometry. We reduce the bilevel problem to a
single-level MILP using strong duality of the inner LP (or solver indicator
constraints). Get this right and unit-tested IN ISOLATION before wiring it into
must_sets.py / optforce.py.

See docs/algorithm.md for the full derivation (primal inner LP -> dual -> strong
duality equality -> single-level MILP).
"""

from __future__ import annotations

import cobra


def build_inner_dual(model: cobra.Model):
    """Construct the dual of the inner stoichiometric LP.

    Returns the dual variables and constraints needed for the strong-duality
    linearisation. Document the correspondence: primal constraint i <-> dual var i.
    """
    raise NotImplementedError


def big_m_from_ranges(min_flux: float, max_flux: float) -> float:
    """Derive a valid big-M from FVA bounds (never an arbitrary constant)."""
    raise NotImplementedError


def solve_bilevel(*args, **kwargs):
    """Assemble and solve the single-level reduction. Backend-specific bits live in
    solvers.py; this orchestrates them."""
    raise NotImplementedError
