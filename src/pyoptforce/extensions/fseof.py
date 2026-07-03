"""FVSEOF: Flux Variability Scanning based on Enforced Objective Flux.

Purpose
-------
Screen a metabolic model for **gene amplification (over-expression) targets** — and,
just as importantly, **reverse-check** a reaction the experimentalist has already
over-expressed against the model. Two ways in:

* *forced increase* — enforce a rising sequence of an objective flux (a product
  exchange, or the biomass reaction itself) and see which internal fluxes are dragged
  up with it. The classic use case (Choi et al. 2010; Park et al. 2012).
* *reverse check via observed growth* — anchor the bottom of the sweep at a measured
  wild-type / observed growth rate and the top at the model's maximal growth, then ask
  whether a specific reaction (the one whose gene was over-expressed in the lab) is
  flagged as an amplification target, and *how strong* the evidence is.

The method is deliberately cheap: LP/FVA only, no bilevel MILP, no Gurobi. It composes
:mod:`pyoptforce.fva`-style FVA calls and the model helpers in
:mod:`pyoptforce.model`; it never touches the core OptForce stages (per the extensions
policy in ``CLAUDE.md``).

The scan and three evidence tiers
---------------------------------
Sweep the *enforced* reaction over ``n_steps`` levels from ``lo`` to ``hi``. At every
level the enforced flux is pinned (``bounds = (level, level)``) and FVA is run over the
scanned reactions, giving ``[v_min(level), v_max(level)]`` for each. Write ``WT`` for the
bottom rung (``levels[0]`` — no/least enforcement) and ``top`` for the top rung
(``levels[-1]`` — maximal enforcement). For each reaction we report three tiers of
increasing logical strength:

1. ``slope`` > 0 — **weak (classic FVSEOF)**. Least-squares slope of the *midpoint*
   ``(v_min+v_max)/2`` vs. enforced level. This is the published criterion and it is
   weak: within ``[v_min, v_max]`` the cell is free to sit anywhere, so a rising midpoint
   only says the feasible *envelope* drifts up — not that any individual flux must rise.

2. ``min_monotone_up`` — **our first strengthening**. The *low end* ``v_min`` (not the
   midpoint) rises monotonically across the sweep. Stronger than a midpoint trend because
   it constrains the worst case, but still permits overlap with the WT range.

3. ``must_up`` — **OptForce-style necessity**. ``v_min(top) > v_max(WT)``: the target-rung
   range is **disjoint above** the WT range, so *every* feasible high-production flux
   exceeds *every* feasible WT flux. This is exactly OptForce's MUSTU test
   (``min_M > max_W``) evaluated across the swept envelope — a genuine necessity
   guarantee, not a correlation. See ``docs/fvseof.md`` for the derivation and the worked
   toy example where these three tiers separate cleanly.

Down-regulation / attenuation targets are the mirror image (``slope < 0``,
``max_monotone_down``, ``must_down = v_max(top) < v_min(WT)``).

The capacity caveat
-------------------
FVSEOF (like OptForce) sees only stoichiometry and flux bounds — never enzyme
abundance. Over-expressing a reaction that is *not* pressed against a binding upper bound
is a no-op in the model. This matters most for *growth as the enforced flux*: in a plain
FBA model growth is already at its optimum, so ``lo`` (observed growth) must genuinely lie
below ``hi`` (model max growth) for the sweep to have headroom. If it does not — i.e. the
observed rate already equals the model maximum — there is nothing to scan and a clear
error is raised. To model an over-expression that lifts growth *above* the current FBA
optimum you need an enzyme-constrained model (GECKO/sMOMENT) or measured limiting bounds;
this module reports what the supplied model can support, no more.

References
----------
* Choi HS, Lee SY, Kim TY, Woo HM (2010). In silico identification of gene amplification
  targets for improvement of lycopene production. *Appl. Environ. Microbiol.* 76(10).
* Park JM, Park HM, Kim WJ, Kim HU, Kim TY, Lee SY (2012). Flux variability scanning based
  on enforced objective flux for identifying gene amplification targets.
  *BMC Syst. Biol.* 6:106.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis

from pyoptforce import model as model_mod

_TOL = 1e-6

# Verdicts, strongest first. Ordering is used for ranking and for `min_verdict` filters.
_UP_VERDICTS = ("must_up", "amplification", "weak_up")
_DOWN_VERDICTS = ("must_down", "attenuation", "weak_down")
_VERDICT_RANK = {v: i for i, v in enumerate(
    ("must_up", "must_down", "amplification", "attenuation", "weak_up", "weak_down", "none")
)}


@dataclass
class FvseofRow:
    """Per-reaction verdict from an FVSEOF scan.

    ``slope`` is the classic (weak) FVSEOF criterion; ``min_monotone_up`` /
    ``max_monotone_down`` are our monotone-worst-case strengthening; ``must_up`` /
    ``must_down`` are the OptForce-style necessity flags (WT/target range disjointness).
    ``verdict`` collapses these into the single strongest applicable label.
    """

    reaction: str
    slope: float
    verdict: str
    must_up: bool
    must_down: bool
    min_monotone_up: bool
    max_monotone_down: bool
    vmin_wt: float
    vmax_wt: float
    vmin_top: float
    vmax_top: float

    @property
    def is_amplification(self) -> bool:
        """True for any up-regulation verdict (weak_up, amplification, or must_up)."""
        return self.verdict in _UP_VERDICTS

    @property
    def is_attenuation(self) -> bool:
        """True for any down-regulation verdict (weak_down, attenuation, or must_down)."""
        return self.verdict in _DOWN_VERDICTS


@dataclass
class FvseofResult:
    """Full result of an FVSEOF scan: per-reaction verdicts plus the raw sweep profiles.

    Every scanned reaction appears in ``rows`` (so :meth:`check` can reverse-look-up an
    arbitrary reaction), and the level-by-level ``[v_min, v_max]`` profiles are retained
    in ``profiles`` for inspection/plotting — nothing is hidden in locals.
    """

    enforced_reaction: str
    levels: list[float]
    rows: list[FvseofRow] = field(default_factory=list)
    # reaction id -> {"vmin": [...], "vmax": [...]} aligned with ``levels``
    profiles: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    # -------------------------------------------------------------- lookups / filters
    def _by_id(self) -> dict[str, FvseofRow]:
        return {r.reaction: r for r in self.rows}

    def check(self, reaction_id: str) -> FvseofRow:
        """Reverse-check a specific reaction (e.g. the lab-over-expressed gene's reaction).

        Raises ``KeyError`` if the reaction was not part of the scan.
        """
        try:
            return self._by_id()[reaction_id]
        except KeyError:
            raise KeyError(
                f"{reaction_id!r} was not scanned; pass it in `reactions=` or scan the "
                "whole model."
            ) from None

    def interpret(self, reaction_id: str) -> str:
        """One-line, human-readable verdict for a reverse-check — for lab comparison."""
        row = self.check(reaction_id)
        enf = self.enforced_reaction
        msgs = {
            "must_up": (
                f"{reaction_id}: NECESSARY up-regulation - its flux range is disjoint "
                f"above the wild-type range as {enf} is forced up, so every feasible "
                "high-state flux exceeds every WT flux. Strong model support for the "
                "observed over-expression effect."
            ),
            "amplification": (
                f"{reaction_id}: amplification target - its worst-case (min) flux rises "
                f"monotonically with {enf}, but the range still overlaps WT, so the "
                "increase is favoured, not forced. Moderate support."
            ),
            "weak_up": (
                f"{reaction_id}: weak/associative up-regulation - only the midpoint of "
                f"its range trends up with {enf}; the low end need not rise. Weak "
                "support (classic FVSEOF signal only)."
            ),
            "must_down": (
                f"{reaction_id}: NECESSARY down-regulation - its range is disjoint below "
                "WT. Over-expressing it opposes the enforced objective."
            ),
            "attenuation": (
                f"{reaction_id}: attenuation target - worst-case (max) flux falls "
                "monotonically. Over-expression is not indicated."
            ),
            "weak_down": (
                f"{reaction_id}: weak down-regulation signal - midpoint trends down only."
            ),
            "none": (
                f"{reaction_id}: no directional signal - the model does not couple this "
                f"reaction to {enf}. If the lab shows an effect, it is likely a "
                "kinetic/expression capacity the (stoichiometric) model does not "
                "represent; consider an enzyme-constrained model."
            ),
        }
        return msgs[row.verdict]

    def _filter(self, verdicts: tuple[str, ...], min_verdict: str | None) -> pd.DataFrame:
        allowed = set(verdicts)
        if min_verdict is not None:
            if min_verdict not in verdicts:
                raise ValueError(
                    f"min_verdict {min_verdict!r} must be one of {verdicts}."
                )
            cutoff = verdicts.index(min_verdict)
            allowed = set(verdicts[: cutoff + 1])
        rows = [r for r in self.rows if r.verdict in allowed]
        return _rows_to_frame(rows)

    def amplification_targets(self, *, min_verdict: str | None = None) -> pd.DataFrame:
        """Up-regulation targets, strongest first.

        ``min_verdict`` sets the weakest tier to include, ordered
        ``must_up`` > ``amplification`` > ``weak_up``; e.g. ``min_verdict="amplification"``
        drops the weak midpoint-only hits. Default: all up-regulation verdicts.
        """
        return self._filter(_UP_VERDICTS, min_verdict)

    def attenuation_targets(self, *, min_verdict: str | None = None) -> pd.DataFrame:
        """Down-regulation / knockdown targets, strongest first (mirror of the above)."""
        return self._filter(_DOWN_VERDICTS, min_verdict)

    def necessary_up(self) -> list[str]:
        """Reactions with the OptForce-style ``must_up`` necessity guarantee."""
        return [r.reaction for r in self.rows if r.must_up]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per scanned reaction, strongest verdict first."""
        return _rows_to_frame(self.rows)


def _rows_to_frame(rows: list[FvseofRow]) -> pd.DataFrame:
    ordered = sorted(
        rows, key=lambda r: (_VERDICT_RANK[r.verdict], -abs(r.slope))
    )
    df = pd.DataFrame(
        [
            {
                "reaction": r.reaction,
                "verdict": r.verdict,
                "slope": r.slope,
                "must_up": r.must_up,
                "must_down": r.must_down,
                "min_monotone_up": r.min_monotone_up,
                "max_monotone_down": r.max_monotone_down,
                "vmin_wt": r.vmin_wt,
                "vmax_wt": r.vmax_wt,
                "vmin_top": r.vmin_top,
                "vmax_top": r.vmax_top,
            }
            for r in ordered
        ],
        columns=[
            "reaction", "verdict", "slope", "must_up", "must_down",
            "min_monotone_up", "max_monotone_down",
            "vmin_wt", "vmax_wt", "vmin_top", "vmax_top",
        ],
    )
    return df.set_index("reaction")


def _classify(
    vmin: np.ndarray, vmax: np.ndarray, levels: np.ndarray, tol: float
) -> tuple[str, bool, bool, bool, bool, float]:
    """Turn one reaction's swept range profile into (verdict, flags, slope)."""
    mids = 0.5 * (vmin + vmax)
    # Least-squares slope of the midpoint vs enforced level (classic FVSEOF signal).
    slope = float(np.polyfit(levels, mids, 1)[0]) if len(levels) > 1 else 0.0

    # Our strengthening: the *worst case* (low end / high end) moves monotonically.
    min_monotone_up = bool(
        np.all(np.diff(vmin) >= -tol) and (vmin[-1] - vmin[0] > tol)
    )
    max_monotone_down = bool(
        np.all(np.diff(vmax) <= tol) and (vmax[0] - vmax[-1] > tol)
    )

    # OptForce-style necessity: top-rung range disjoint from the WT (bottom) range.
    must_up = bool(vmin[-1] > vmax[0] + tol)
    must_down = bool(vmax[-1] < vmin[0] - tol)

    if must_up:
        verdict = "must_up"
    elif must_down:
        verdict = "must_down"
    elif min_monotone_up and slope > tol:
        verdict = "amplification"
    elif max_monotone_down and slope < -tol:
        verdict = "attenuation"
    elif slope > tol:
        verdict = "weak_up"
    elif slope < -tol:
        verdict = "weak_down"
    else:
        verdict = "none"

    return verdict, must_up, must_down, min_monotone_up, max_monotone_down, slope


def _baseline_flux(
    model: cobra.Model, reaction_id: str, biomass_reaction: str | None
) -> float:
    """Flux through ``reaction_id`` in the wild-type max-growth FBA solution.

    Used as the default bottom of the sweep. If ``biomass_reaction`` is given the
    objective is anchored to it (so 'wild type' means max growth, not whatever the SBML
    shipped); otherwise the model's current objective is used.
    """
    with model as m:
        if biomass_reaction is not None:
            model_mod.set_linear_objective(m, biomass_reaction, "max")
        sol = m.optimize()
    if sol.status != "optimal":
        raise ValueError(
            f"Wild-type FBA is not optimal (status={sol.status!r}); cannot set the "
            "sweep baseline. Provide `lo` explicitly."
        )
    return float(sol.fluxes[reaction_id])


def fvseof(
    model: cobra.Model,
    enforced_reaction: str,
    *,
    biomass_reaction: str | None = None,
    lo: float | None = None,
    hi: float | None = None,
    n_steps: int = 10,
    reactions: list[str] | None = None,
    biomass_floor_fraction: float = 0.0,
    tol: float = _TOL,
) -> FvseofResult:
    """Run an FVSEOF over-expression-target scan.

    Parameters
    ----------
    model:
        A cobra model. Not mutated (every change is inside a ``with model:`` context).
    enforced_reaction:
        The objective flux to force upward across the sweep — a product exchange
        (classic amplification screen) **or** the biomass reaction (growth as target /
        reverse-check via observed growth rate).
    biomass_reaction:
        Optional. Anchors the wild-type baseline to maximal growth and, if
        ``biomass_floor_fraction > 0``, keeps a viable-growth floor at every swept level.
        Ignored when it equals ``enforced_reaction`` (growth is already pinned then).
    lo, hi:
        Bottom and top enforced-flux levels. ``hi`` defaults to the theoretical maximum
        of ``enforced_reaction``; ``lo`` defaults to its wild-type (max-growth) flux. For
        the *reverse-check via observed growth* use case, pass ``lo=observed_growth_rate``
        and let ``hi`` default to the model's maximal growth. Requires ``lo < hi`` with
        real headroom — see the capacity caveat in the module docstring.
    n_steps:
        Number of enforced levels (inclusive of ``lo`` and ``hi``). Must be >= 2.
    reactions:
        Restrict the scan (and FVA cost) to these reaction ids; default is every
        reaction. The enforced reaction is always excluded from the results.
    biomass_floor_fraction:
        If > 0 and ``biomass_reaction`` is given (and differs from the enforced
        reaction), keep biomass >= this fraction of max growth at every level so the scan
        only explores viable states (mirrors classic FVSEOF). Default 0 (full
        stoichiometric range, matching OptForce stage-2 FVA).

    Returns
    -------
    FvseofResult
        Per-reaction verdicts (:meth:`FvseofResult.to_dataframe`,
        :meth:`amplification_targets`) plus :meth:`check` / :meth:`interpret` for
        reverse-checking a specific reaction.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 (need a bottom and a top rung).")
    model_mod.require_reactions(model, [enforced_reaction])
    if biomass_reaction is not None:
        model_mod.require_reactions(model, [biomass_reaction])
    if not 0.0 <= biomass_floor_fraction <= 1.0:
        raise ValueError("biomass_floor_fraction must be in [0, 1].")

    if hi is None:
        hi = model_mod.theoretical_max(model, enforced_reaction)
    if lo is None:
        lo = _baseline_flux(model, enforced_reaction, biomass_reaction)

    if hi - lo <= tol:
        raise ValueError(
            f"No headroom to scan: lo={lo:.4g} >= hi={hi:.4g} for {enforced_reaction!r}. "
            "For growth as the enforced flux this means the observed/baseline rate "
            "already equals the model's maximum, so a purely stoichiometric model has "
            "nothing to force. Pass an explicit lo below hi (e.g. a measured growth "
            "rate), or use an enzyme-constrained model that has capacity headroom."
        )

    # Growth floor is only meaningful when biomass is a *separate* reaction from the
    # enforced one (otherwise growth is already pinned to the swept level).
    apply_floor = (
        biomass_floor_fraction > 0.0
        and biomass_reaction is not None
        and biomass_reaction != enforced_reaction
    )
    min_growth = 0.0
    if apply_floor:
        max_growth = model_mod.theoretical_max(model, biomass_reaction)
        min_growth = biomass_floor_fraction * max_growth

    scan_ids = list(reactions) if reactions is not None else [
        r.id for r in model.reactions
    ]
    if reactions is not None:
        model_mod.require_reactions(model, scan_ids)
    scan_ids = [r for r in scan_ids if r != enforced_reaction]
    if not scan_ids:
        raise ValueError("Nothing to scan (only the enforced reaction was selected).")

    levels = np.linspace(lo, hi, n_steps)

    # profiles[rid] = {"vmin": [...], "vmax": [...]} aligned with successful levels
    profiles: dict[str, dict[str, list[float]]] = {
        rid: {"vmin": [], "vmax": []} for rid in scan_ids
    }
    used_levels: list[float] = []

    for level in levels:
        with model as m:
            if biomass_reaction is not None and biomass_reaction != enforced_reaction:
                model_mod.set_linear_objective(m, biomass_reaction, "max")
                if apply_floor:
                    bio = m.reactions.get_by_id(biomass_reaction)
                    bio.lower_bound = min(min_growth, bio.upper_bound)
            m.reactions.get_by_id(enforced_reaction).bounds = (float(level), float(level))
            try:
                # processes=1: we call FVA once per swept level, so per-call process-pool
                # spawn overhead is pure loss (and on Windows a pool re-imports the
                # caller's __main__, which is fragile). Sequential is faster here.
                df = flux_variability_analysis(
                    m, reaction_list=scan_ids, fraction_of_optimum=0.0, processes=1
                )
            except cobra.exceptions.OptimizationError:
                # This enforced level is infeasible under the current floor/bounds; the
                # envelope simply isn't defined there, so skip the rung rather than fail.
                continue
        used_levels.append(float(level))
        for rid, row in df.iterrows():
            profiles[rid]["vmin"].append(float(row["minimum"]))
            profiles[rid]["vmax"].append(float(row["maximum"]))

    if len(used_levels) < 2:
        raise ValueError(
            "Fewer than two enforced levels were feasible; cannot detect a trend. "
            "Lower biomass_floor_fraction or narrow [lo, hi] to a feasible sub-range."
        )

    lvl = np.asarray(used_levels)
    rows: list[FvseofRow] = []
    for rid in scan_ids:
        vmin = np.asarray(profiles[rid]["vmin"])
        vmax = np.asarray(profiles[rid]["vmax"])
        verdict, must_up, must_down, min_mono, max_mono, slope = _classify(
            vmin, vmax, lvl, tol
        )
        rows.append(
            FvseofRow(
                reaction=rid,
                slope=slope,
                verdict=verdict,
                must_up=must_up,
                must_down=must_down,
                min_monotone_up=min_mono,
                max_monotone_down=max_mono,
                vmin_wt=float(vmin[0]),
                vmax_wt=float(vmax[0]),
                vmin_top=float(vmin[-1]),
                vmax_top=float(vmax[-1]),
            )
        )

    return FvseofResult(
        enforced_reaction=enforced_reaction,
        levels=used_levels,
        rows=rows,
        profiles={rid: profiles[rid] for rid in scan_ids},
    )
