"""
Parameters, reaction network and Antimony / SBML export.

The model in one paragraph. A filament carries N nucleation sites. A site
catches a k_seed-nt exact match, on the homologous donor with probability
f and somewhere in the genome background otherwise, then extends it one
nucleotide at a time. The next nucleotide pairs with probability p, which
is 1 - delta on the donor and p_het in the background. A mismatch is
stepped over while fewer than max_mismatches have been met and blocks the
joint for good after that; a blocked joint can only fall off. Reaching
L_commit is the point of no return.

    nucleate   S             -> T[k, 0]       kon f   (donor)
                                              kon (1-f) (background)
    zip        T[L, j]       -> T[L+1, j]     kext p
    step over  T[L, j < m]   -> T[L+1, j+1]   kext (1-p)
    block      T[L, m]       -> Tb[L]         kext (1-p)
    fall off   T[L, j], Tb[L] -> S            koff(L)

Every reaction is first order and the sites share nothing, so the model is
N independent copies of one small Markov chain. theory.py builds that
chain from the same reaction list and solves it exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace, asdict

MODEL_NAME = "homology_zipper"
SIMULATION_KEYS = ("t_end", "n_points", "n_cells", "seed")


# =====================================================================
# Parameters
# =====================================================================

@dataclass(frozen=True)
class Params:
    """Every number of the model. Immutable; `with_` returns a copy."""

    # Structural
    n_sites: int = 200                       # nucleation sites on the filament
    f: float = 1.57e-3                       # share of seeds landing on the donor
    k_seed: int = 8                          # seed length, nt
    l_commit: int = 30                       # commitment length, nt
    max_mismatches: int = 0                  # mismatches stepped over before blocking

    # Kinetic, per minute (kext per nucleotide per minute)
    kon: float = 0.006
    kext: float = 300.0
    koff0: float = 0.7
    lam: float | None = None                 # koff decay length, nt; None = flat koff

    # Sequence: a scalar, or one value per step L = k_seed .. l_commit - 1
    p_hom: float | tuple[float, ...] = 1.0
    p_het: float | tuple[float, ...] = 0.271

    # Simulation
    t_end: float = 480.0
    n_points: int = 481
    n_cells: int = 2000
    seed: int = 1999

    def __post_init__(self) -> None:
        if self.k_seed < 1:
            raise ValueError("k_seed must be >= 1 nt")
        if self.l_commit <= self.k_seed:
            raise ValueError("l_commit must be strictly greater than k_seed")
        if self.max_mismatches < 0:
            raise ValueError("max_mismatches must be >= 0")
        if not 0.0 <= self.f <= 1.0:
            raise ValueError("f must lie in [0, 1]")
        for name in ("kon", "kext", "koff0"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.lam is not None and self.lam <= 0.0:
            raise ValueError("lam must be > 0, or None for a flat koff")
        for name in ("p_hom", "p_het"):
            # A scalar becomes a float and a profile a tuple, so that the
            # rest of the code can rely on isinstance(p, float).
            p = getattr(self, name)
            p = float(p) if isinstance(p, (int, float)) else tuple(float(v) for v in p)
            object.__setattr__(self, name, p)
            if isinstance(p, tuple) and len(p) != self.n_steps:
                raise ValueError(
                    f"{name} must be a scalar or a profile of {self.n_steps} values, "
                    f"one per step L = {self.k_seed}..{self.l_commit - 1}; got {len(p)}")
            if any(not 0.0 <= v <= 1.0 for v in ((p,) if isinstance(p, float) else p)):
                raise ValueError(f"{name} must lie in [0, 1]")

    # ---- Derived ----

    @property
    def n_steps(self) -> int:
        """n = l_commit - k_seed, the number of nucleotides still to check."""
        return self.l_commit - self.k_seed

    @property
    def lengths(self) -> range:
        """The zipping states, L = k_seed .. l_commit - 1."""
        return range(self.k_seed, self.l_commit)

    def p(self, track: str, l: int) -> float:
        """Match probability of the next nucleotide, on track H or X."""
        p = self.p_hom if track == "H" else self.p_het
        return p if isinstance(p, float) else p[l - self.k_seed]

    def koff(self, l: int) -> float:
        """Fall-off rate of a joint holding L paired nucleotides."""
        if self.lam is None:
            return self.koff0
        return self.koff0 * math.exp(-(l - self.k_seed) / self.lam)

    def with_(self, **changes) -> "Params":
        return replace(self, **changes)

    # ---- I/O ----

    @classmethod
    def from_config(cls, cfg: dict) -> "Params":
        """Build from the params.yaml keys, which allow a few aliases."""
        alias = {"N": "n_sites", "L_commit": "l_commit", "seed_zero": "seed"}
        kwargs = {alias.get(key, key): value for key, value in cfg.items()}
        if "delta" in kwargs:
            if "p_hom" in kwargs:
                raise ValueError("give either 'delta' or 'p_hom', not both")
            kwargs["p_hom"] = 1.0 - float(kwargs.pop("delta"))
        if kwargs.get("lam") == 0:
            kwargs["lam"] = None
        unknown = set(kwargs) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        return cls(**kwargs)

    def uid(self) -> int:
        """Identifier of the model. Simulation settings do not enter it."""
        cfg = {k: v for k, v in asdict(self).items() if k not in SIMULATION_KEYS}
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        return int(digest, 16) % 2**32


# =====================================================================
# Reaction network
# =====================================================================

@dataclass(frozen=True)
class Reaction:
    src: str
    dst: str
    law: str       # Antimony rate expression, without the species factor
    rate: float    # its value under the Params it was built from


@dataclass
class Network:
    """The single-site reaction list. Species are in declaration order,
    which is also the column order of the generator matrix."""
    params: list[tuple[str, float]]
    species: list[str]
    reactions: list[Reaction]
    absorbing: tuple[str, ...] = ("RH", "RX")


def state_name(track: str, l: int, j: int = 0) -> str:
    """A zipping joint: track H or X, L paired nt, j mismatches stepped over."""
    return f"{track}{l}" if j == 0 else f"{track}{l}m{j}"


def blocked_name(track: str, l: int) -> str:
    return f"{track}b{l}"


def _p_exprs(prm: Params, track: str, l: int) -> tuple[str, str]:
    # A scalar p stays a model parameter, so it can be tuned on a loaded
    # model without rebuilding. A measured profile is written out value by
    # value, since there is nothing to tune.
    p = prm.p_hom if track == "H" else prm.p_het
    symbol = "p_hom" if track == "H" else "p_het"
    if isinstance(p, float):
        return symbol, f"(1 - {symbol})"
    v = p[l - prm.k_seed]
    return repr(v), repr(1.0 - v)


def _koff_expr(prm: Params, l: int) -> str:
    if prm.lam is None:
        return "koff0"
    return f"koff0 * exp(-({l} - {prm.k_seed}) / lam)"


def build_network(prm: Params) -> Network:
    """Both the Antimony source and the generator matrix of theory.py are
    built from this list, so they cannot drift apart."""
    m = prm.max_mismatches
    lengths = list(prm.lengths)
    tracks = (("H", "RH"), ("X", "RX"))

    params: list[tuple[str, float]] = [
        ("N", prm.n_sites), ("f", prm.f),
        ("kon", prm.kon), ("kext", prm.kext), ("koff0", prm.koff0),
    ]
    if prm.lam is not None:
        params.append(("lam", prm.lam))
    if isinstance(prm.p_hom, float):
        params.append(("p_hom", prm.p_hom))
    if isinstance(prm.p_het, float):
        params.append(("p_het", prm.p_het))

    species: list[str] = ["S"]
    for track, _ in tracks:
        for j in range(m + 1):
            species.extend(state_name(track, l, j) for l in lengths)
        species.extend(blocked_name(track, l) for l in lengths)
    species.extend(("RH", "RX"))

    rx = [
        Reaction("S", state_name("H", prm.k_seed), "kon * f", prm.kon * prm.f),
        Reaction("S", state_name("X", prm.k_seed), "kon * (1 - f)",
                 prm.kon * (1.0 - prm.f)),
    ]
    for track, committed in tracks:
        for j in range(m + 1):
            for l in lengths:
                src = state_name(track, l, j)
                p = prm.p(track, l)
                zip_f, mis_f = _p_exprs(prm, track, l)
                nxt = l + 1

                # Pairs: one more nucleotide, mismatch count unchanged.
                dst = state_name(track, nxt, j) if nxt < prm.l_commit else committed
                rx.append(Reaction(src, dst, f"kext * {zip_f}", prm.kext * p))

                # Mismatch: stepped over while j < m, blocking at j = m.
                if j < m:
                    dst = (state_name(track, nxt, j + 1) if nxt < prm.l_commit
                           else committed)
                else:
                    dst = blocked_name(track, l)
                rx.append(Reaction(src, dst, f"kext * {mis_f}", prm.kext * (1.0 - p)))

                rx.append(Reaction(src, "S", _koff_expr(prm, l), prm.koff(l)))

        for l in lengths:
            rx.append(Reaction(blocked_name(track, l), "S", _koff_expr(prm, l),
                               prm.koff(l)))

    return Network(params=params, species=species, reactions=rx)


# =====================================================================
# Antimony and SBML
# =====================================================================

def to_antimony(net: Network, prm: Params) -> str:
    out: list[str] = []
    out.append(f"// {MODEL_NAME} (homozip): the mismatch-limited zipper\n")
    out.append(f"// uid {prm.uid()}  k_seed {prm.k_seed}  L_commit {prm.l_commit}"
               f"  max_mismatches {prm.max_mismatches}\n")
    out.append(f"model {MODEL_NAME}()\n")
    out.append("    compartment cell = 1;\n\n")

    out.append("    // ---- Parameters ----\n")
    for name, value in net.params:
        out.append(f"    {name} = {value};\n")
    out.append("\n")

    out.append("    // ---- Species ----\n")
    out.append("    species S in cell;    // free nucleation sites\n")
    for track, label in (("H", "homologous donor"), ("X", "genome background")):
        for j in range(prm.max_mismatches + 1):
            names = ", ".join(state_name(track, l, j) for l in prm.lengths)
            tag = f"{label}, {j} mismatch(es) stepped over" if prm.max_mismatches \
                else label
            out.append(f"    species {names} in cell;    // zipping, {tag}\n")
        names = ", ".join(blocked_name(track, l) for l in prm.lengths)
        out.append(f"    species {names} in cell;    // blocked, {label}\n")
    out.append("    species RH, RX in cell;    // committed, L reached L_commit\n\n")

    out.append("    // ---- Initial state ----\n")
    out.append(f"    S = {prm.n_sites};\n")
    for name in net.species[1:]:
        out.append(f"    {name} = 0;\n")
    out.append("\n")

    out.append("    // ---- Reactions ----\n")
    for i, r in enumerate(net.reactions, start=1):
        out.append(f"    R{i}: {r.src} -> {r.dst}; ({r.law}) * {r.src};\n")
    out.append("end\n")
    return "".join(out)


def export_sbml(antimony: str) -> str:
    import tellurium as te
    return te.antimonyToSBML(antimony)
