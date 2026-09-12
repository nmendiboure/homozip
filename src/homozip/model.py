"""
The mismatch-limited zipper: one site, one Markov chain, solved exactly.

A filament carries N nucleation sites. A site catches a k_seed-nt exact
match, on the homologous donor with probability f and somewhere in the
genome background otherwise, then extends it one nucleotide at a time.
The donor is one locus in a nucleus and is within reach of the filament
only a fraction c of the time; the background is everywhere. That single
number is the only place where the nucleus enters.
The next nucleotide pairs with probability p, which is 1 - delta on the
donor and p_het in the background. A mismatch is stepped over while fewer
than max_mismatches have been met and blocks the joint for good after
that; a blocked joint can only fall off. Reaching L_commit is the point
of no return.

    nucleate   S             -> T[k, 0]       kon f c   (donor)
                                              kon (1-f) (background)
    zip        T[L, j]       -> T[L+1, j]     kext p
    step over  T[L, j < m]   -> T[L+1, j+1]   kext (1-p)
    block      T[L, m]       -> Tb[L]         kext (1-p)
    fall off   T[L, j], Tb[L] -> S            koff

Every transition is first order and the sites share nothing, so the
filament is N independent copies of this chain. The whole solution rests
on one observation: at any length the three rates add up to kext + koff,
which does not contain p. The sequence decides which branch is taken,
never how long the joint waits.

Section 1: parameters. Section 2: the chain. Section 3: what can be
written with a pen (commitment probability, fidelity bound, occupancy,
commitment rate). Section 4: the exact first-passage distribution.

Notation: k = k_seed, Lc = l_commit, n = Lc - k, m = max_mismatches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm
from scipy.special import comb


# =====================================================================
# 1. Parameters
# =====================================================================

@dataclass(frozen=True)
class Params:
    """Every number of the model. Immutable; `with_` returns a copy."""

    # Structural
    n_sites: int = 200                       # nucleation sites on the filament
    f: float = 1.57e-3                       # share of seeds landing on the donor
    contact: float = 0.025                   # share of the time the donor is within reach
    k_seed: int = 8                          # seed length, nt
    l_commit: int = 30                       # commitment length, nt
    max_mismatches: int = 0                  # mismatches stepped over before blocking

    # Kinetic, per minute (kext per nucleotide per minute)
    kon: float = 1.0
    kext: float = 300.0
    koff0: float = 0.7
    lam: float | None = None                 # koff decay length, nt; None = flat koff

    # Sequence: a scalar, or one value per step L = k_seed .. l_commit - 1
    p_hom: float | tuple[float, ...] = 1.0
    p_het: float | tuple[float, ...] = 0.271

    def __post_init__(self) -> None:
        if self.k_seed < 1:
            raise ValueError("k_seed must be >= 1 nt")
        if self.l_commit <= self.k_seed:
            raise ValueError("l_commit must be strictly greater than k_seed")
        if self.max_mismatches < 0:
            raise ValueError("max_mismatches must be >= 0")
        for name in ("f", "contact"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
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

    @classmethod
    def from_config(cls, cfg: dict) -> "Params":
        """Build from the params.yaml keys, which allow a few aliases."""
        alias = {"N": "n_sites", "L_commit": "l_commit"}
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


# =====================================================================
# 2. The chain
# =====================================================================

@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    rate: float


def state_name(track: str, l: int, j: int = 0) -> str:
    """A zipping joint: track H or X, L paired nt, j mismatches stepped over."""
    return f"{track}{l}" if j == 0 else f"{track}{l}m{j}"


def blocked_name(track: str, l: int) -> str:
    return f"{track}b{l}"


def build_chain(prm: Params) -> tuple[list[str], list[Transition]]:
    """States in declaration order, and every transition between them.
    RH and RX, the donor and background commitments, are absorbing."""
    m = prm.max_mismatches
    lengths = list(prm.lengths)
    tracks = (("H", "RH"), ("X", "RX"))

    states: list[str] = ["S"]
    for track, _ in tracks:
        for j in range(m + 1):
            states.extend(state_name(track, l, j) for l in lengths)
        states.extend(blocked_name(track, l) for l in lengths)
    states.extend(("RH", "RX"))

    rx = [Transition("S", state_name("H", prm.k_seed), prm.kon * prm.f * prm.contact),
          Transition("S", state_name("X", prm.k_seed), prm.kon * (1.0 - prm.f))]
    for track, committed in tracks:
        for j in range(m + 1):
            for l in lengths:
                src, p, nxt = state_name(track, l, j), prm.p(track, l), l + 1
                # Pairs: one more nucleotide, mismatch count unchanged.
                dst = state_name(track, nxt, j) if nxt < prm.l_commit else committed
                rx.append(Transition(src, dst, prm.kext * p))
                # Mismatch: stepped over while j < m, blocking at j = m.
                if j < m:
                    dst = (state_name(track, nxt, j + 1) if nxt < prm.l_commit
                           else committed)
                else:
                    dst = blocked_name(track, l)
                rx.append(Transition(src, dst, prm.kext * (1.0 - p)))
                rx.append(Transition(src, "S", prm.koff(l)))
        for l in lengths:
            rx.append(Transition(blocked_name(track, l), "S", prm.koff(l)))
    return states, rx


def generator(prm: Params) -> tuple[np.ndarray, dict[str, int]]:
    """Generator matrix of the chain, and the index of each state."""
    states, rx = build_chain(prm)
    idx = {s: i for i, s in enumerate(states)}
    q = np.zeros((len(states), len(states)))
    for r in rx:
        i, j = idx[r.src], idx[r.dst]
        q[i, j] += r.rate
        q[i, i] -= r.rate
    return q, idx


# =====================================================================
# 3. Pen and paper
# =====================================================================

def exit_rate(prm: Params, l: int) -> float:
    return prm.kext + prm.koff(l)


def step_probabilities(prm: Params, track: str, l: int) -> tuple[float, float, float]:
    """(pairs, mismatch, falls off) at length L. They sum to one."""
    r = exit_rate(prm, l)
    p = prm.p(track, l)
    return prm.kext * p / r, prm.kext * (1.0 - p) / r, prm.koff(l) / r


def reach_probabilities(prm: Params, track: str = "H") -> dict[tuple[int, int], float]:
    """pi(L, j): chance that a seed ever occupies L paired nt with j
    mismatches behind it. Entries at L = Lc are committed."""
    pi = {(prm.k_seed, 0): 1.0}
    for l in prm.lengths:
        z, x, _ = step_probabilities(prm, track, l)
        for j in range(prm.max_mismatches + 1):
            w = pi.get((l, j), 0.0)
            if w == 0.0:
                continue
            pi[(l + 1, j)] = pi.get((l + 1, j), 0.0) + w * z
            if j < prm.max_mismatches:
                pi[(l + 1, j + 1)] = pi.get((l + 1, j + 1), 0.0) + w * x
    return pi


def commit_probability(prm: Params, track: str = "H") -> float:
    """Chance that one seed reaches L_commit."""
    pi = reach_probabilities(prm, track)
    return sum(pi.get((prm.l_commit, j), 0.0) for j in range(prm.max_mismatches + 1))


def commit_probability_closed_form(prm: Params, p: float) -> float:
    """The same thing with a pen, for a scalar p and a flat koff:

        P = Q * Prob[ Binomial(n, 1 - p) <= m ],   Q = (kext / (kext + koff0))^n

    which is p^n Q when no mismatch is tolerated. Q is where the rates
    enter, and Q does not depend on p. That is why kinetics cancel out of
    any ratio of commit probabilities.
    """
    if prm.lam is not None:
        raise ValueError("this form assumes a flat koff")
    n, m = prm.n_steps, prm.max_mismatches
    q = (prm.kext / (prm.kext + prm.koff0)) ** n
    tail = sum(comb(n, i, exact=True) * (1.0 - p) ** i * p ** (n - i)
               for i in range(m + 1))
    return q * tail


def kinetic_yield(prm: Params) -> float:
    """Q: the part of the commit probability that the sequence does not touch."""
    return commit_probability(prm.with_(p_hom=1.0), "H")


def discrimination(prm: Params, p_a: float, p_b: float) -> float:
    """Odds of committing on donor A rather than donor B. Free of rates."""
    return (commit_probability(prm.with_(p_hom=p_a))
            / commit_probability(prm.with_(p_hom=p_b)))


def false_commitment_odds(prm: Params) -> float:
    """Background commitments per donor commitment. Background seeds
    outnumber donor seeds (1 - f) / (f c) to one, and each is less likely
    to get through."""
    return ((1.0 - prm.f) / (prm.f * prm.contact)
            * commit_probability(prm, "X") / commit_probability(prm, "H"))


def min_commitment_steps(prm: Params, epsilon: float = 1e-3) -> float:
    """Shortest test that keeps false commitments below a fraction epsilon
    of the true ones:

        n >= ln( (1 - f) / (f c epsilon) ) / ln( p_hom / p_het )

    No rate appears in it. With c = 1 it is a statement about the sequence
    alone; a donor within reach a fraction c of the time costs a further
    ln(1/c) / ln(p_hom/p_het) nucleotides. For m = 0 and scalar p.
    """
    if not (isinstance(prm.p_hom, float) and isinstance(prm.p_het, float)):
        raise ValueError("this bound needs scalar p_hom and p_het")
    return (math.log((1.0 - prm.f) / (prm.f * prm.contact * epsilon))
            / math.log(prm.p_hom / prm.p_het))


def occupancy_time(prm: Params, track: str) -> float:
    """How long one nucleation keeps its site busy, on average. Each visit
    to a length costs 1 / (kext + koff); a joint that blocks there waits a
    further 1 / koff. That second term is what makes a background seed
    expensive: it blocks after about 1 / (1 - p_het) nucleotides and then
    sits on its site for 1 / koff0 rather than for 1 / kext."""
    pi = reach_probabilities(prm, track)
    m = prm.max_mismatches
    tau = 0.0
    for l in prm.lengths:
        _, x, _ = step_probabilities(prm, track, l)
        tau += sum(pi.get((l, j), 0.0) for j in range(m + 1)) / exit_rate(prm, l)
        tau += pi.get((l, m), 0.0) * x / prm.koff(l)
    return tau


def free_sites(prm: Params) -> float:
    """Free sites at steady state, N / (1 + kon tau), by Little's law."""
    tau = (prm.f * prm.contact * occupancy_time(prm, "H")
           + (1.0 - prm.f) * occupancy_time(prm, "X"))
    return prm.n_sites / (1.0 + prm.kon * tau)


def occupied_fraction(prm: Params) -> float:
    """Share of the N sites holding a joint at steady state."""
    return 1.0 - free_sites(prm) / prm.n_sites


def commit_rate(prm: Params) -> float:
    """Donor commitments per minute, for the whole filament."""
    return prm.kon * prm.f * prm.contact * free_sites(prm) * commit_probability(prm, "H")


def transit_time_mean(prm: Params) -> float:
    """Time to zip from k to L_commit, given that it works. Independent of
    p: a divergent donor is not slower, only less likely to finish."""
    return sum(1.0 / exit_rate(prm, l) for l in prm.lengths)


def mean_search_time(prm: Params) -> float:
    """Steady-state estimate of the time to the first donor commitment.
    mean_search_time_exact is the real thing; they differ by one transit."""
    rate = commit_rate(prm)
    return math.inf if rate <= 0.0 else 1.0 / rate + transit_time_mean(prm)


def search_time_decomposition(prm: Params) -> dict[str, float]:
    """The search time split the way Bitran et al. (2017) write it,

        <T> = (1 / P_commit) [ tau_diff + tau_off + tau_target ] / N

    with tau_diff = 1 / (kon f c) the wait for a donor encounter, tau_off the
    time lost on the background between two of them, (1-f)/(f c) blocked
    joints of tau_X each, and tau_target the time spent on the donor
    itself. The total is exactly 1 / commit_rate.
    """
    p_commit = commit_probability(prm, "H")
    tau_diff = 1.0 / (prm.kon * prm.f * prm.contact)
    tau_off = (1.0 - prm.f) / (prm.f * prm.contact) * occupancy_time(prm, "X")
    tau_target = occupancy_time(prm, "H")
    per_site = (tau_diff + tau_off + tau_target) / p_commit
    return {"rounds": 1.0 / p_commit, "tau_diff": tau_diff, "tau_off": tau_off,
            "tau_target": tau_target, "per_site": per_site,
            "parallel": per_site / prm.n_sites}


def optimal_koff0(prm: Params) -> float:
    """Best koff0, from the small-koff0 expansion of the commit rate:

        koff0* = ( -c + sqrt(c^2 + 4 c kext / n_eff) ) / 2,  c = (1 - p_het) kon

    Below it, blocked background seeds clog the filament; above it, donor
    seeds are released mid-zip. Under weak clogging it is the geometric
    mean sqrt(c kext / n_eff), where n_eff is the nucleotides' worth of
    koff a joint has to survive (n when koff is flat).
    """
    n_eff = (float(prm.n_steps) if prm.lam is None
             else sum(math.exp(-i / prm.lam) for i in range(prm.n_steps)))
    c = (1.0 - float(np.mean(prm.p_het))) * prm.kon
    return 0.5 * (-c + math.sqrt(c * c + 4.0 * c * prm.kext / n_eff))


# =====================================================================
# 4. Exact first passage
# =====================================================================
# Absorption into RH starting from a free site is a phase-type
# distribution on the chain above. With N independent sites,
#
#     F_N(t) = 1 - (1 - F_1(t))^N,   f_N(t) = N (1 - F_1)^(N-1) f_1(t),
#
# computed on a uniform grid by one matrix exponential and one
# vector-matrix product per time point. No ensemble, no approximation.

def first_passage(prm: Params, t, start: str | None = None) -> dict[str, np.ndarray]:
    """One site: cdf of committing on the donor and on the background, the
    density, and the hazard. The hazard settles to a constant once the site
    has forgotten where it started, which is the memoryless statement made
    quantitative. The grid must be uniform and start at 0."""
    t = np.asarray(t, dtype=float)
    if t.size < 2 or t[0] != 0.0 or not np.allclose(np.diff(t), t[1] - t[0]):
        raise ValueError("the time grid must be uniform and start at 0")
    q, idx = generator(prm)
    step = expm(q * (t[1] - t[0]))

    v = np.zeros(q.shape[0])
    v[idx[start or "S"]] = 1.0
    q_rh = q[:, idx["RH"]]

    cdf_h, cdf_x, pdf_h = np.empty(t.size), np.empty(t.size), np.empty(t.size)
    for i in range(t.size):
        cdf_h[i], cdf_x[i], pdf_h[i] = v[idx["RH"]], v[idx["RX"]], v @ q_rh
        v = v @ step

    surv = np.clip(1.0 - cdf_h, 1e-300, None)
    return {"t": t, "cdf_H": cdf_h, "cdf_X": cdf_x, "pdf_H": pdf_h,
            "hazard_H": pdf_h / surv}


def search_time_distribution(prm: Params, t) -> dict[str, np.ndarray]:
    """First donor commitment among N independent sites: exact cdf and pdf."""
    fp = first_passage(prm, t)
    surv = 1.0 - fp["cdf_H"]
    n = prm.n_sites
    return {"t": fp["t"], "cdf": 1.0 - surv ** n,
            "pdf": n * surv ** (n - 1) * fp["pdf_H"], "single": fp}


def mean_search_time_exact(prm: Params, t_max: float | None = None,
                           n_points: int = 4001) -> float:
    """Integral of (1 - F_1)^N, closed beyond t_max by the flat hazard."""
    if t_max is None:
        t_max = 6.0 * mean_search_time(prm)
        if not math.isfinite(t_max):
            return math.inf
    t = np.linspace(0.0, t_max, n_points)
    fp = first_passage(prm, t)
    surv_n = (1.0 - fp["cdf_H"]) ** prm.n_sites
    h = float(fp["hazard_H"][-1])
    tail = surv_n[-1] / (prm.n_sites * h) if h > 0.0 else math.inf
    return float(np.trapz(surv_n, t)) + tail


def half_time(prm: Params, n_points: int = 2001) -> float:
    """Time at which half of the cells have committed on the donor, read
    off the exact distribution. Below ln 2 times the mean on a crowded
    filament, because a fresh filament starts with every site free."""
    t_max = 4.0 * mean_search_time(prm)
    if not math.isfinite(t_max):
        return math.inf
    t = np.linspace(0.0, t_max, n_points)
    cdf = search_time_distribution(prm, t)["cdf"]
    if cdf[-1] < 0.5:
        return math.inf
    return float(np.interp(0.5, cdf, t))


# =====================================================================
# Summary
# =====================================================================

def summarize(prm: Params, exact: bool = True) -> dict:
    p_h, p_x = commit_probability(prm, "H"), commit_probability(prm, "X")
    split = search_time_decomposition(prm)
    out = {
        "n_steps": prm.n_steps,
        "max_mismatches": prm.max_mismatches,
        "p_hom": prm.p_hom if isinstance(prm.p_hom, float) else "profile",
        "p_het": prm.p_het if isinstance(prm.p_het, float) else "profile",
        "Q": kinetic_yield(prm),
        "P_commit_hom": p_h,
        "P_commit_het": p_x,
        "discrimination": p_h / p_x if p_x > 0 else math.inf,
        "false_per_true": false_commitment_odds(prm),
        "tau_hom": occupancy_time(prm, "H"),
        "tau_het": occupancy_time(prm, "X"),
        "S_free": free_sites(prm),
        "transit_time": transit_time_mean(prm),
        "commit_rate": commit_rate(prm),
        "tau_diff": split["tau_diff"],
        "tau_off": split["tau_off"],
        "tau_target": split["tau_target"],
        "mean_search_time": mean_search_time(prm),
        "koff0_opt": optimal_koff0(prm),
    }
    if exact:
        out["mean_search_time_exact"] = mean_search_time_exact(prm)
        out["half_time"] = half_time(prm)
    if prm.max_mismatches == 0 and isinstance(prm.p_hom, float) \
            and isinstance(prm.p_het, float):
        out["n_min_fidelity_1e-3"] = min_commitment_steps(prm, 1e-3)
    return out
