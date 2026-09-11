"""
The figures. One plot per file, PDF only, nothing drawn inside anything
else.

    divergence.pdf          the divergence law, and what a tolerance does to it
    filament_stability.pdf  the best filament stability
    commitment_length.pdf   what a longer test costs, and what it buys
    first_passage.pdf       first passage, exact against Gillespie
    hazard.pdf              the single-site hazard: the transit is the only memory
    spectrum_survival.pdf   the match-length spectrum behind f
    spectrum_hazard.pdf     the measured p(L) behind p_het
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from .model import Params                # noqa: E402
from . import theory, simulate, genome   # noqa: E402

BLUE, RED, ORANGE, GREY, LIGHT = "#1f5fa9", "#c0392b", "#e8992a", "#7f7f7f", "#5b8fc7"


def _scalar(prm: Params) -> Params:
    # Panels that sweep L_commit cannot carry a profile of fixed length, so
    # a measured p(L) is replaced by its mean there.
    changes = {key: float(np.mean(getattr(prm, key)))
               for key in ("p_hom", "p_het")
               if not isinstance(getattr(prm, key), float)}
    return prm.with_(**changes) if changes else prm


# =====================================================================
# Panels
# =====================================================================

def panel_divergence(ax, prm: Params, n_rep: int, seed: int) -> None:
    prm = _scalar(prm)
    n = prm.n_steps
    deltas = np.linspace(0.0, 0.20, 80)

    for m, color in ((0, BLUE), (1, ORANGE), (2, RED)):
        curve = [theory.commit_probability(prm.with_(max_mismatches=m, p_hom=1.0 - d))
                 for d in deltas]
        label = rf"theory, $m = {m}$"
        if m == 0:
            label += r"  ($P = (1-\delta)^n Q$)"
        ax.semilogy(deltas, curve, color=color, lw=2, label=label)

    q = theory.kinetic_yield(prm)
    ax.semilogy(deltas, q * np.exp(-n * deltas), ls="--", color=GREY, lw=1.2,
                label=rf"$Q\,e^{{-n\delta}}$,  $n = {n}$")

    for m, color, marker in ((0, BLUE, "o"), (2, RED, "s")):
        xs, ys, es = [], [], []
        for d in (0.0, 0.03, 0.06, 0.10, 0.15, 0.20):
            mean, err, _ = simulate.isolated_seed_commit_fraction(
                prm.with_(max_mismatches=m, p_hom=1.0 - d), n_rep, seed)
            xs.append(d)
            ys.append(max(mean, 1e-6))
            es.append(err)
        ax.errorbar(xs, ys, yerr=es, fmt=marker, color=color, ms=5, capsize=3,
                    mfc="white", label=f"SSA, $m = {m}$, {n_rep} lone seeds")

    ax.text(0.97, 0.05,
            f"background, $p_{{het}} = {prm.p_het:.3f}$:  "
            f"$P = {theory.commit_probability(prm, 'X'):.1e}$\n"
            rf"discrimination $= (p_{{hom}}/p_{{het}})^n = "
            rf"{theory.discrimination(prm, 1.0, prm.p_het):.1e}$",
            transform=ax.transAxes, fontsize=7, color=RED, ha="right", va="bottom")

    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"donor divergence $\delta$")
    ax.set_ylabel(r"$P_{\mathrm{commit}}$ per seed")
    ax.set_title("The divergence law, and the shoulder a tolerance adds",
                 fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, ls="--", alpha=0.4)


def panel_koff_optimum(ax, prm: Params) -> None:
    grid = prm.kext * np.logspace(-5, 0, 300)
    for mult, color in ((1, BLUE), (30, ORANGE), (300, RED)):
        local = prm.with_(kon=prm.kon * mult)
        rate = np.asarray([theory.commit_rate(local.with_(koff0=ko)) for ko in grid])
        ax.loglog(grid / prm.kext, rate / rate.max(), color=color, lw=1.8,
                  label=rf"$k_{{on}}N = {local.kon * prm.n_sites:.3g}$ min$^{{-1}}$")
        ko_star, _ = theory.optimal_koff0_numeric(local)
        ax.axvline(ko_star / prm.kext, color=color, ls=":", lw=1.0)

    ax.axvline(prm.koff0 / prm.kext, color="black", lw=1.4)
    ax.text(prm.koff0 / prm.kext * 0.8, 1.6, "this calibration",
            fontsize=7, rotation=90, ha="right", va="top")
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"$k_{off,0} / k_{ext}$   (dotted: best value for each $k_{on}$)")
    ax.set_ylabel("commit rate (normalised)")
    ax.set_title("How stable the filament should be", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def panel_commit_length(ax, prm: Params) -> None:
    prm = _scalar(prm)
    lcs = np.arange(prm.k_seed + 2, 71)

    for delta, color in ((0.0, BLUE), (0.02, ORANGE), (0.05, RED)):
        t_mean = [theory.mean_search_time_qss(
            prm.with_(l_commit=int(lc), p_hom=1.0 - delta)) for lc in lcs]
        ax.semilogy(lcs, t_mean, color=color, lw=1.8,
                    label=rf"search time, $\delta = {delta}$")

    ax2 = ax.twinx()
    odds = [theory.false_commitment_odds(prm.with_(l_commit=int(lc))) for lc in lcs]
    ax2.semilogy(lcs, odds, color="black", ls=":", lw=1.6, label="false / true (right)")
    for eps in (1e-2, 1e-4):
        ax2.axhline(eps, color=GREY, lw=0.8, ls="--")
        ax2.text(lcs[-1], eps * 1.3, rf"$\epsilon = {eps:g}$", fontsize=6.5,
                 ha="right", color=GREY)
    ax2.set_ylabel("false commitments per true one (dotted)")
    ax2.set_ylim(1e-30, 1e3)

    # The shortest test on the swept grid that meets one false commitment per
    # thousand. Derived from f and p_het alone, so it needs no outside value.
    marks = [(prm.l_commit, r"$L_{commit}$")]
    bound = next((int(lc) for lc, o in zip(lcs, odds) if o <= 1e-3), None)
    if bound is not None and bound != prm.l_commit:
        marks.insert(0, (bound, "fidelity bound\n$\\epsilon = 10^{-3}$"))
    for lc, label in marks:
        ax.axvline(lc, color="black", lw=1.0, alpha=0.6)
        ax.text(lc + 0.5, ax.get_ylim()[0] * 1.5, label, fontsize=6.5, va="bottom")

    ax.set_xlabel(r"commitment length $L_{commit}$ (nt)")
    ax.set_ylabel("mean search time (min)")
    ax.set_title("What a longer test costs, and what it buys", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.grid(True, ls="--", alpha=0.4)


def panel_first_passage(ax, prm: Params, n_cells: int, seed: int) -> dict:
    ens = simulate.run_ensemble(prm, n_cells, seed)
    t = ens["t"]
    times = ens["first_commit"]
    committed = times[~np.isnan(times)]
    dist = theory.search_time_distribution(prm, t)
    rate = theory.commit_rate(prm)

    n_bins = 30
    width = t[-1] / n_bins
    if committed.size:
        ax.hist(committed, bins=n_bins, range=(0, t[-1]),
                weights=np.full(committed.size, 1.0 / (times.size * width)),
                alpha=0.55, color=LIGHT,
                label=f"SSA, {committed.size}/{times.size} cells by {t[-1]:.0f} min")
    ax.plot(t, dist["pdf"], color=RED, lw=2,
            label=f"exact, N = {prm.n_sites} independent sites")
    ax.plot(t, rate * np.exp(-rate * t), ls="--", color="black", lw=1.2,
            label=r"memoryless, $\lambda = k_{on} f S_{free} P_{commit}$")
    ax.set_xlabel("time of the first donor commitment (min)")
    ax.set_ylabel("density (per cell)")
    ax.set_title("First passage: exact against Gillespie", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)
    return {"ensemble": ens, "exact": dist}


def panel_hazard(ax, prm: Params) -> None:
    """The single-site hazard, on the seconds scale. It rises over one
    transit time and is flat after that, which is the whole memory of the
    search. first_passage.pdf is the same statement on the minutes scale."""
    rate = theory.commit_rate(prm)
    transit = theory.transit_time_mean(prm)
    t_s = np.linspace(0.0, 360.0 * transit, 400)
    fp = theory.first_passage(prm, t_s / 60.0)

    ax.plot(t_s, prm.n_sites * fp["hazard_H"] / rate, color=RED, lw=2)
    ax.axhline(1.0, color="black", ls="--", lw=1.0,
               label=r"$\lambda = k_{on} f S_{free} P_{commit}$")
    ax.axvline(60.0 * transit, color=GREY, lw=1.0, ls=":")
    ax.annotate(f"one transit, {60 * transit:.1f} s",
                xy=(60.0 * transit, 0.30), xytext=(60.0 * transit * 1.4, 0.22),
                fontsize=7.5, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, t_s[-1])
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$N\,h_1(t)\;/\;\lambda$")
    ax.set_title("The transit is the only memory of the search", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, ls="--", alpha=0.4)


# =====================================================================
# The two spectrum panels
# =====================================================================

def panel_spectrum_survival(ax, agg: dict, bg: dict, l_commit: int) -> None:
    k = agg["ksize"]
    L_a, S_a = np.array(agg["L"], float), np.array(agg["survival"])
    L_b, S_b = np.array(bg["L"], float), np.array(bg["survival"])
    f_donor = genome.donor_fraction(agg, l_commit)
    p_het = genome.background_p(bg)

    ax.semilogy(L_a, np.where(S_a > 0, S_a, np.nan), "o-", ms=3.5, lw=1.6,
                color=BLUE, label="measured, whole genome")
    ax.semilogy(L_b, np.where(S_b > 0, S_b, np.nan), "s-", ms=3.5, lw=1.6,
                color=RED, label="measured, donor masked (the background)")
    ax.semilogy(L_a, 0.25 ** (L_a - k), ":", lw=1.4, color=GREY,
                label=r"random null  $0.25^{\,L-k}$")
    ax.semilogy(L_a, p_het ** (L_a - k), "--", lw=1.4, color=ORANGE,
                label=rf"composition null  ${p_het:.3f}^{{\,L-k}}$")
    ax.semilogy(L_a, (1 - f_donor) * p_het ** (L_a - k) + f_donor, "-", lw=1.0,
                color="black", alpha=0.6,
                label=r"two tracks  $(1-f)\,p_{het}^{L-k} + f$")
    ax.axhline(f_donor, color=BLUE, lw=1.0, alpha=0.4)
    ax.annotate(f"plateau: the donor alone" + "\n" + f"$f$ = {f_donor:.2e}",
                xy=(24, f_donor * 2.2), fontsize=7.5, color=BLUE)
    ax.axvline(l_commit, color="black", lw=1.2)
    ax.annotate(r"$L_{commit}$", xy=(l_commit - 0.6, 3e-6), fontsize=8,
                rotation=90, ha="right")
    ax.set_ylim(1e-7, 2)
    ax.set_xlim(k, 34)
    ax.set_xlabel("L  (matched nt)")
    ax.set_ylabel(r"$P(M \geq L)$  per (site, locus) pair")
    ax.set_title("Match-length spectrum, LY filament against S288c", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def panel_spectrum_hazard(ax, agg: dict, bg: dict) -> None:
    k = agg["ksize"]
    p_het = genome.background_p(bg)
    hL_a, h_a = np.array(agg["hazard_L"], float), np.array(agg["hazard"])
    hL_b, h_b = np.array(bg["hazard_L"], float), np.array(bg["hazard"])
    ok = h_b > 0

    ax.plot(hL_a, h_a, "o-", ms=3.5, lw=1.6, color=BLUE, label="whole genome")
    ax.plot(hL_b[ok], h_b[ok], "s-", ms=3.5, lw=1.6, color=RED,
            label="background only (donor masked)")
    ax.axhline(0.25, ls=":", lw=1.4, color=GREY)
    ax.axhline(p_het, ls="--", lw=1.4, color=ORANGE)
    ax.annotate("0.25  random null", xy=(21.5, 0.205), fontsize=7.5, color=GREY)
    ax.annotate(f"{p_het:.3f}  genome composition", xy=(18.4, 0.283),
                fontsize=7.5, color=ORANGE)
    ax.annotate("the donor takes over" + "\n" + "the surviving population",
                xy=(13.3, 0.62), xytext=(14.8, 0.74), fontsize=7.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.annotate("repeats" + "\n" + "(Ty, paralogues)",
                xy=(18.1, 0.44), xytext=(19.6, 0.58), fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_ylim(0, 1.08)
    ax.set_xlim(k, 24)
    ax.set_xlabel("L  (matched nt)")
    ax.set_ylabel(r"$p(L) = P(M \geq L{+}1)\,/\,P(M \geq L)$")
    ax.set_title(r"The model's $p$, measured", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, ls="--", alpha=0.35)


# =====================================================================
# Figures: one plot per file
# =====================================================================

def _plot(outdir: str, name: str, draw, figsize=(7.0, 5.2)) -> str:
    """One axes, one PDF, named after what it shows."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw(ax)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_model(prm: Params, outdir: str, quick: bool = False) -> list[str]:
    n_rep = 300 if quick else 4000
    n_cells = 100 if quick else prm.n_cells
    return [
        _plot(outdir, "divergence",
              lambda ax: panel_divergence(ax, prm, n_rep, prm.seed)),
        _plot(outdir, "filament_stability",
              lambda ax: panel_koff_optimum(ax, prm)),
        _plot(outdir, "commitment_length",
              lambda ax: panel_commit_length(ax, prm)),
        _plot(outdir, "first_passage",
              lambda ax: panel_first_passage(ax, prm, n_cells, prm.seed)),
        _plot(outdir, "hazard",
              lambda ax: panel_hazard(ax, prm)),
    ]


def figure_spectrum(genome_json: str, background_json: str, outdir: str,
                    l_commit: int = 30) -> list[str]:
    agg = genome.load(genome_json)
    bg = genome.load(background_json)
    return [
        _plot(outdir, "spectrum_survival",
              lambda ax: panel_spectrum_survival(ax, agg, bg, l_commit)),
        _plot(outdir, "spectrum_hazard",
              lambda ax: panel_spectrum_hazard(ax, agg, bg)),
    ]
