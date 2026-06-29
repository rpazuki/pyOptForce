"""Shared fixtures: a tiny network whose MUST and FORCE sets are known by hand.

Network (carbon flows left to right)::

    up_A: -> A        (uptake, capacity 10)
    r1:  A -> B       branch to growth
    r2:  A -> C       branch to product
    bio: B ->         biomass (objective)
    prod: C -> P
    EX_P: P ->        product exchange (target)

Steady state forces  up_A = r1 + r2,  bio = r1,  EX_P = prod = r2.

* Max growth (WT basal): all carbon to B  -> bio = r1 = 10, r2 = prod = EX_P = 0.
* Theoretical max product: all carbon to C -> EX_P = 10.  threshold @0.5 = 5.

Hand-derived expectations (WT at max growth vs target EX_P >= 5):
  MUSTU (must increase): r2, prod, EX_P     (0 in WT, >=5 in target)
  MUSTL (must decrease): r1, bio            (10 in WT, <=5 in target)
  FORCE k=1 valid: up-regulate r2 / prod / EX_P (forces flux into product).
  NOT valid k=1: down-regulate r1 / bio alone (caps growth but lets uptake drop,
  so product is not forced).
"""

from __future__ import annotations

import cobra
import pytest


@pytest.fixture
def toy_model() -> cobra.Model:
    m = cobra.Model("toy")
    A = cobra.Metabolite("A")
    B = cobra.Metabolite("B")
    C = cobra.Metabolite("C")
    P = cobra.Metabolite("P")
    m.add_metabolites([A, B, C, P])

    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = cobra.Reaction(rid)
        r.lower_bound = lb
        r.upper_bound = ub
        r.add_metabolites(stoich)
        return r

    m.add_reactions([
        rxn("up_A", {A: 1.0}, lb=0.0, ub=10.0),
        rxn("r1", {A: -1.0, B: 1.0}),
        rxn("r2", {A: -1.0, C: 1.0}),
        rxn("bio", {B: -1.0}),
        rxn("prod", {C: -1.0, P: 1.0}),
        rxn("EX_P", {P: -1.0}),
    ])
    m.objective = "bio"
    return m
