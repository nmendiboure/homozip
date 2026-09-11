"""
Closed forms, and the exact first-passage solution.

The whole model rests on one observation. A joint sitting at length L can
do three things, and the rates add up to

    kext p + kext (1 - p) + koff(L) = kext + koff(L),

which does not contain p. The time spent at each length is therefore the
same whatever the sequence; only the branch taken differs. Section 1 is
what follows from that with a pen. Section 2 solves the chain exactly: the
N sites are independent copies of one Markov chain, so the search time is
the minimum of N phase-type variables, and the Gillespie runs of
simulate.py only have to agree.

Notation: k = k_seed, Lc = l_commit, n = Lc - k, m = max_mismatches.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm
from scipy.special import comb
from scipy.stats import gamma as gamma_dist

from .model import Params, Network, build_network


# =====================================================================
# 1. Pen and paper
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
    """The same thing for a scalar p and a flat koff:

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
    """Odds of committing on donor A rather than donor B."""
    return (commit_probability(prm.with_(p_hom=p_a))
            / commit_probability(prm.with_(p_hom=p_b)))


def false_commitment_odds(prm: Params) -> float:
    """Background commitments per donor commitment. Background seeds
    outnumber donor seeds (1 - f) / f to one, and each is less likely to
    get through."""
    return ((1.0 - prm.f) / prm.f
            * commit_probability(prm, "X") / commit_probability(prm, "H"))


def specificity(prm: Params) -> float:
    """Fraction of commitments that land on the donor."""
    return 1.0 / (1.0 + false_commitment_odds(prm))


def min_commitment_steps(prm: Params, epsilon: float = 1e-3) -> float:
    """Shortest test that keeps false commitments below a fraction epsilon
    of the true ones:

        n >= ln( (1 - f) / (f epsilon) ) / ln( p_hom / p_het )

    No rate appears in it, and nothing was fitted. For m = 0 and scalar p.
    """
    if not (isinstance(prm.p_hom, float) and isinstance(prm.p_het, float)):
        raise ValueError("this bound needs scalar p_hom and p_het")
    return math.log((1.0 - prm.f) / (prm.f * epsilon)) / math.log(prm.p_hom / prm.p_het)


# ---- Occupancy and the commit rate ----

def occupancy_time(prm: Params, track: str) -> float:
    """How long one nucleation keeps its site busy, on average.

    Each visit to a length costs 1 / (kext + koff); a joint that blocks
    there waits a further 1 / koff. That second term is what makes a
    background seed expensive: it blocks after about 1 / (1 - p_het)
    nucleotides and then sits on its site for 1 / koff0 rather than for
    1 / kext.
    """
    pi = reach_probabilities(prm, track)
    m = prm.max_mismatches
    tau = 0.0
    for l in prm.lengths:
        r = exit_rate(prm, l)
        _, x, _ = step_probabilities(prm, track, l)
        tau += sum(pi.get((l, j), 0.0) for j in range(m + 1)) / r
        tau += pi.get((l, m), 0.0) * x / prm.koff(l)
    return tau


def free_sites(prm: Params) -> float:
    """Free sites at steady state, by Little's law."""
    tau = prm.f * occupancy_time(prm, "H") + (1.0 - prm.f) * occupancy_time(prm, "X")
    return prm.n_sites / (1.0 + prm.kon * tau)


def commit_rate(prm: Params) -> float:
    return prm.kon * prm.f * free_sites(prm) * commit_probability(prm, "H")


def transit_time_mean(prm: Params) -> float:
    """Time to zip from k to L_commit, given that it works. Independent of
    p: a divergent donor is not slower, only less likely to finish."""
    return sum(1.0 / exit_rate(prm, l) for l in prm.lengths)


def mean_search_time_qss(prm: Params) -> float:
    """Steady-state estimate. mean_search_time_exact is the real thing."""
    rate = commit_rate(prm)
    return math.inf if rate <= 0.0 else 1.0 / rate + transit_time_mean(prm)


def search_time_decomposition(prm: Params) -> dict[str, float]:
    """The search time split the way Bitran et al. (2017) write it,

        <T> = (1 / P_commit) [ tau_diff + tau_off + tau_target ] / N

    with tau_diff = 1 / (kon f) the wait for a donor encounter, tau_off the
    time lost on the background between two of them, tau_target the time
    spent on the donor itself. The total is exactly 1 / commit_rate.
    """
    p_commit = commit_probability(prm, "H")
    tau_diff = 1.0 / (prm.kon * prm.f)
    tau_off = (1.0 - prm.f) / prm.f * occupancy_time(prm, "X")
    tau_target = occupancy_time(prm, "H")
    per_site = (tau_diff + tau_off + tau_target) / p_commit
    return {"rounds": 1.0 / p_commit, "tau_diff": tau_diff, "tau_off": tau_off,
            "tau_target": tau_target, "per_site": per_site,
            "parallel": per_site / prm.n_sites}


def lam_eff(prm: Params) -> float:
    """Nucleotides' worth of koff a joint has to survive. Equals n when
    koff is flat."""
    if prm.lam is None:
        return float(prm.n_steps)
    return sum(math.exp(-i / prm.lam) for i in range(prm.n_steps))


def optimal_koff0_approx(prm: Params) -> float:
    """Best koff0, from the small-koff0 expansion of the commit rate:

        koff0* = ( -c + sqrt(c^2 + 4 c kext / lam_eff) ) / 2,  c = (1 - p_het) kon

    Below it, blocked background seeds clog the filament; above it, donor
    seeds are released mid-zip. Under weak clogging it is the geometric
    mean sqrt(c kext / lam_eff).
    """
    c = (1.0 - float(np.mean(prm.p_het))) * prm.kon
    return 0.5 * (-c + math.sqrt(c * c + 4.0 * c * prm.kext / lam_eff(prm)))


def optimal_koff0_numeric(prm: Params, decades=(-5.0, 1.0), n_points=2000):
    """Scan koff0 on a log grid; return (koff0*, commit rate there)."""
    best = (0.0, 0.0)
    lo, hi = decades
    for i in range(n_points):
        koff0 = prm.kext * 10.0 ** (lo + (hi - lo) * i / (n_points - 1))
        rate = commit_rate(prm.with_(koff0=koff0))
        if rate > best[1]:
            best = (koff0, rate)
    return best


# =====================================================================
# 2. Exact first passage
# =====================================================================
# The single-site chain has about 4n(m+1) states and its generator comes
# from the very reaction list that produces the Antimony source. Absorption
# into RH starting from a free site is a phase-type distribution. With N
# independent sites,
#
#     F_N(t) = 1 - (1 - F_1(t))^N,   f_N(t) = N (1 - F_1)^(N-1) f_1(t),
#
# computed on a uniform grid by one matrix exponential and one
# vector-matrix product per time point. No ensemble, no approximation.

def generator(net: Network) -> tuple[np.ndarray, dict[str, int]]:
    """Generator matrix and the species index, in declaration order."""
    idx = {s: i for i, s in enumerate(net.species)}
    n = len(net.species)
    q = np.zeros((n, n))
    for r in net.reactions:
        i, j = idx[r.src], idx[r.dst]
        q[i, j] += r.rate
        q[i, i] -= r.rate
    return q, idx


def _check_uniform(t: np.ndarray) -> float:
    if t.size < 2 or t[0] != 0.0:
        raise ValueError("the time grid must start at 0 and have at least 2 points")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("the time grid must be uniform")
    return dt


def first_passage(prm: Params, t, start: str | None = None) -> dict[str, np.ndarray]:
    """One site: cdf of committing on the donor and on the background, the
    density, and the hazard. The hazard settles to a constant once the site
    has forgotten where it started, which is the memoryless statement made
    quantitative."""
    net = build_network(prm)
    q, idx = generator(net)
    t = np.asarray(t, dtype=float)
    step = expm(q * _check_uniform(t))

    v = np.zeros(len(net.species))
    v[idx[start or "S"]] = 1.0
    q_rh = q[:, idx["RH"]]

    cdf_h = np.empty(t.size)
    cdf_x = np.empty(t.size)
    pdf_h = np.empty(t.size)
    for i in range(t.size):
        cdf_h[i] = v[idx["RH"]]
        cdf_x[i] = v[idx["RX"]]
        pdf_h[i] = v @ q_rh
        v = v @ step

    surv = np.clip(1.0 - cdf_h, 1e-300, None)
    return {"t": t, "cdf_H": cdf_h, "cdf_X": cdf_x, "pdf_H": pdf_h,
            "hazard_H": pdf_h / surv}


def search_time_distribution(prm: Params, t, n_sites: int | None = None):
    """First donor commitment among N independent sites: exact cdf and pdf."""
    n = prm.n_sites if n_sites is None else n_sites
    fp = first_passage(prm, t)
    surv = 1.0 - fp["cdf_H"]
    return {"t": fp["t"], "cdf": 1.0 - surv ** n,
            "pdf": n * surv ** (n - 1) * fp["pdf_H"], "single": fp}


def mean_search_time_exact(prm: Params, n_sites: int | None = None,
                           t_max: float | None = None, n_points: int = 4001) -> float:
    """Integral of (1 - F_1)^N, closed beyond t_max by the flat hazard."""
    n = prm.n_sites if n_sites is None else n_sites
    if t_max is None:
        t_max = 6.0 * mean_search_time_qss(prm)
        if not math.isfinite(t_max):
            return math.inf
    t = np.linspace(0.0, t_max, n_points)
    fp = first_passage(prm, t)
    surv_n = (1.0 - fp["cdf_H"]) ** n
    h = float(fp["hazard_H"][-1])
    tail = surv_n[-1] / (n * h) if h > 0.0 else math.inf
    return float(np.trapz(surv_n, t)) + tail


def transit_time_pdf(prm: Params, t) -> np.ndarray:
    """Density of the k -> L_commit transit, given success. Erlang for a
    flat koff, phase-type otherwise, and the same whatever p is."""
    t = np.asarray(t, dtype=float)
    rates = [exit_rate(prm, l) for l in prm.lengths]
    if prm.lam is None:
        return gamma_dist.pdf(t, a=len(rates), scale=1.0 / rates[0])
    n = len(rates)
    q = np.zeros((n + 1, n + 1))
    for i, r in enumerate(rates):
        q[i, i] = -r
        q[i, i + 1] = r
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    return np.array([(e0 @ expm(q * ti))[:n] @ q[:n, n] for ti in t])


# =====================================================================
# Summary
# =====================================================================

def summarize(prm: Params, exact: bool = True) -> dict:
    p_h = commit_probability(prm, "H")
    p_x = commit_probability(prm, "X")
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
        "specificity": specificity(prm),
        "tau_hom": occupancy_time(prm, "H"),
        "tau_het": occupancy_time(prm, "X"),
        "S_free": free_sites(prm),
        "transit_time": transit_time_mean(prm),
        "commit_rate": commit_rate(prm),
        "rounds": split["rounds"],
        "tau_diff": split["tau_diff"],
        "tau_off": split["tau_off"],
        "tau_target": split["tau_target"],
        "mean_search_time_qss": mean_search_time_qss(prm),
        "koff0_opt_approx": optimal_koff0_approx(prm),
    }
    if prm.max_mismatches == 0 and isinstance(prm.p_hom, float) \
            and isinstance(prm.p_het, float):
        out["n_min_fidelity_1e-3"] = min_commitment_steps(prm, 1e-3)
    if exact:
        out["mean_search_time_exact"] = mean_search_time_exact(prm)
    return out
