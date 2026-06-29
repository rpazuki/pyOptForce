"""Result containers for the OptForce pipeline.

These are deliberately plain dataclasses so intermediate results stay inspectable
and easy to serialise. Stage code populates them; the OptForce driver holds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FluxRanges:
    """Output of stages 1 & 2: FVA ranges for WT and target-constrained models.

    All four dicts are keyed by reaction id. ``_w`` = wild type (stage 1),
    ``_m`` = target/overproducing mutant network (stage 2).
    """

    min_w: dict[str, float] = field(default_factory=dict)   # minFluxesW
    max_w: dict[str, float] = field(default_factory=dict)   # maxFluxesW
    min_m: dict[str, float] = field(default_factory=dict)   # minFluxesM (target)
    max_m: dict[str, float] = field(default_factory=dict)   # maxFluxesM (target)

    def reactions(self) -> list[str]:
        """Reaction ids present in both the WT and target range sets, ordered."""
        return [r for r in self.min_w if r in self.min_m]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per reaction, columns min_w/max_w/min_m/max_m."""
        rows = {
            r: {
                "min_w": self.min_w.get(r),
                "max_w": self.max_w.get(r),
                "min_m": self.min_m.get(r),
                "max_m": self.max_m.get(r),
            }
            for r in self.min_w
        }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "reaction"
        return df[["min_w", "max_w", "min_m", "max_m"]]


@dataclass
class MustSets:
    """Output of stage 3: reactions whose flux must change to reach the target."""

    mustU: list[str] = field(default_factory=list)          # must increase (1st order)
    mustL: list[str] = field(default_factory=list)          # must decrease (1st order)
    mustUU: list[tuple[str, str]] = field(default_factory=list)
    mustLL: list[tuple[str, str]] = field(default_factory=list)
    mustUL: list[tuple[str, str]] = field(default_factory=list)

    def all_reactions(self) -> set[str]:
        """Every reaction mentioned in any (first- or second-order) MUST set."""
        out: set[str] = set(self.mustU) | set(self.mustL)
        for pairs in (self.mustUU, self.mustLL, self.mustUL):
            for a, b in pairs:
                out.add(a)
                out.add(b)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Long-format table: one row per MUST entry with its set label."""
        rows: list[dict] = []
        for r in self.mustU:
            rows.append({"set": "MUSTU", "order": 1, "reactions": (r,)})
        for r in self.mustL:
            rows.append({"set": "MUSTL", "order": 1, "reactions": (r,)})
        for label, pairs in (("MUSTUU", self.mustUU),
                             ("MUSTLL", self.mustLL),
                             ("MUSTUL", self.mustUL)):
            for pair in pairs:
                rows.append({"set": label, "order": 2, "reactions": tuple(pair)})
        return pd.DataFrame(rows, columns=["set", "order", "reactions"])


@dataclass
class ForceSets:
    """Output of stage 4: enumerated minimal intervention sets."""

    sets: list[dict] = field(default_factory=list)
    # each entry: {"reactions": [...], "type": {rxn: "up"|"down"|"ko"},
    #              "objective": float}

    def __len__(self) -> int:
        return len(self.sets)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per enumerated FORCE set."""
        rows: list[dict] = []
        for i, s in enumerate(self.sets):
            rows.append({
                "solution": i,
                "reactions": tuple(s.get("reactions", ())),
                "type": s.get("type", {}),
                "objective": s.get("objective"),
            })
        return pd.DataFrame(rows, columns=["solution", "reactions", "type", "objective"])
