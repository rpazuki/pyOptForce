"""Result containers for the OptForce pipeline.

These are deliberately plain dataclasses so intermediate results stay inspectable
and easy to serialise. Stage code populates them; the OptForce driver holds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FluxRanges:
    """Output of stages 1 & 2: FVA ranges for WT and target-constrained models."""

    min_w: dict[str, float] = field(default_factory=dict)   # minFluxesW
    max_w: dict[str, float] = field(default_factory=dict)   # maxFluxesW
    min_m: dict[str, float] = field(default_factory=dict)   # minFluxesM (target)
    max_m: dict[str, float] = field(default_factory=dict)   # maxFluxesM (target)

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class MustSets:
    """Output of stage 3: reactions whose flux must change to reach the target."""

    mustU: list[str] = field(default_factory=list)          # must increase (1st order)
    mustL: list[str] = field(default_factory=list)          # must decrease (1st order)
    mustUU: list[tuple[str, str]] = field(default_factory=list)
    mustLL: list[tuple[str, str]] = field(default_factory=list)
    mustUL: list[tuple[str, str]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class ForceSets:
    """Output of stage 4: enumerated minimal intervention sets."""

    sets: list[dict] = field(default_factory=list)
    # each entry: {"reactions": [...], "type": {rxn: "up"|"down"|"ko"}, "objective": float}

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError
