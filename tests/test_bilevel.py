"""Isolation tests for the bilevel engine (roadmap step 3).

The strong-duality reduction must be verified on a trivial LP BEFORE it is relied on
by the higher-order MUST sets / FORCE step.
"""

import numpy as np

from pyoptforce import bilevel


def test_big_m_from_ranges():
    assert bilevel.big_m_from_ranges(-3.0, 5.0, buffer=0.0) == 5.0
    assert bilevel.big_m_from_ranges(-7.0, 2.0, buffer=1.0) == 8.0


def test_strong_duality_selftest():
    primal, dual = bilevel.strong_duality_selftest()
    assert np.isclose(primal, dual, atol=1e-6)
    assert np.isclose(primal, 2.0, atol=1e-6)  # v1=v2=1 -> objective 2
