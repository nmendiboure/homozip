"""
The figures.

model.png     A  the divergence law, and what tolerated mismatches do to it
              B  the best filament stability
              C  what a longer test costs in speed and buys in stringency
              D  first passage, exact against Gillespie
spectrum.png  the genome measurement behind p_het and f
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
        curve = [theory.commit_vs_divergence(prm.with_(max_mismatches=m), d)
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
    ax.set_title("A — The divergence law, and the shoulder a tolerance adds",
                 fontsize=10)
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
    ax.text(prm.koff0 / prm.kext * 0.8, 1.6, r"SHERPA $r_{off1}/r_{elong}$",
            fontsize=7, rotation=90, ha="right", va="top")
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"$k_{off,0} / k_{ext}$   (dotted: best value for each $k_{on}$)")
    ax.set_ylabel("commit rate (normalised)")
    ax.set_title("B — How stable the filament should be", fontsize=10)
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

    for lc, label in ((18, "SHERPA\ndloop_min_size"), (prm.l_commit, r"$L_{commit}$")):
        ax.axvline(lc, color="black", lw=1.0, alpha=0.6)
        ax.text(lc + 0.5, ax.get_ylim()[0] * 1.5, label, fontsize=6.5, va="bottom")

    ax.set_xlabel(r"commitment length $L_{commit}$ (nt)")
    ax.set_ylabel("mean search time (min)")
    ax.set_title("C — What a longer test costs, and what it buys", fontsize=10)
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
    ax.set_title("D — First passage: exact against Gillespie", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)

    # Inset: the single-site hazard, on the seconds scale. It rises over one
    # transit time and is flat after that, which is the whole memory of the
    # search.
    transit = theory.transit_time_mean(prm)
    t_in = np.linspace(0.0, 360.0 * transit, 400)
    fp = theory.first_passage(prm, t_in / 60.0)
    ins = ax.inset_axes([0.55, 0.35, 0.4, 0.3])
    ins.plot(t_in, prm.n_sites * fp["hazard_H"] / rate, color=RED, lw=1.4)
    ins.axhline(1.0, color="black", ls="--", lw=0.8)
    ins.axvline(60.0 * transit, color=GREY, lw=0.8, ls=":")
    ins.set_xlabel("t (s)", fontsize=6.5)
    ins.set_ylabel(r"$N h_1(t)\,/\,\lambda$", fontsize=6.5)
    ins.set_title(f"the {60 * transit:.0f} s transit is the only memory", fontsize=6.5)
    ins.tick_params(labelsize=6)
    ins.set_ylim(0, 1.2)
    return {"ensemble": ens, "exact": dist}


# =====================================================================
# Figures
# =====================================================================

def figure_model(prm: Params, outdir: str, quick: bool = False) -> str:
    n_rep = 300 if quick else 4000
    n_cells = 100 if quick else prm.n_cells
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    panel_divergence(axs[0, 0], prm, n_rep, prm.seed)
    panel_koff_optimum(axs[0, 1], prm)
    panel_commit_length(axs[1, 0], prm)
    panel_first_passage(axs[1, 1], prm, n_cells, prm.seed)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "model.png")
    fig.savefig(out, dpi=160)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out


def figure_spectrum(genome_json: str, background_json: str, outdir: str,
                    l_commit: int = 30) -> str:
    agg = genome.load(genome_json)
    bg = genome.load(background_json)
    k = agg["ksize"]
    L_a, S_a = np.array(agg["L"], float), np.array(agg["survival"])
    L_b, S_b = np.array(bg["L"], float), np.array(bg["survival"])
    f_donor = genome.donor_fraction(agg, l_commit)
    p_het = genome.background_p(bg)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax0.semilogy(L_a, np.where(S_a > 0, S_a, np.nan), "o-", ms=3.5, lw=1.6,
                 color=BLUE, label="measured, whole genome")
    ax0.semilogy(L_b, np.where(S_b > 0, S_b, np.nan), "s-", ms=3.5, lw=1.6,
                 color=RED, label="measured, donor masked (the background)")
    ax0.semilogy(L_a, 0.25 ** (L_a - k), ":", lw=1.4, color=GREY,
                 label=r"random null  $0.25^{\,L-k}$")
    ax0.semilogy(L_a, p_het ** (L_a - k), "--", lw=1.4, color=ORANGE,
                 label=rf"composition null  ${p_het:.3f}^{{\,L-k}}$")
    ax0.semilogy(L_a, (1 - f_donor) * p_het ** (L_a - k) + f_donor, "-", lw=1.0,
                 color="black", alpha=0.6,
                 label=r"two tracks  $(1-f)\,p_{het}^{L-k} + f$")
    ax0.axhline(f_donor, color=BLUE, lw=1.0, alpha=0.4)
    ax0.annotate(f"plateau: the donor alone\n$f$ = {f_donor:.2e}",
                 xy=(24, f_donor * 2.2), fontsize=7.5, color=BLUE)
    ax0.axvline(l_commit, color="black", lw=1.2)
    ax0.annotate(r"$L_{commit}$", xy=(l_commit - 0.6, 3e-6), fontsize=8,
                 rotation=90, ha="right")
    ax0.set_ylim(1e-7, 2)
    ax0.set_xlim(k, 34)
    ax0.set_xlabel("L  (matched nt)")
    ax0.set_ylabel(r"$P(M \geq L)$  per (site, locus) pair")
    ax0.set_title("Match-length spectrum, LY filament against S288c", fontsize=10)
    ax0.legend(fontsize=7.5, loc="lower left")
    ax0.grid(True, which="both", ls="--", alpha=0.35)

    hL_a, h_a = np.array(agg["hazard_L"], float), np.array(agg["hazard"])
    hL_b, h_b = np.array(bg["hazard_L"], float), np.array(bg["hazard"])
    ok = h_b > 0
    ax1.plot(hL_a, h_a, "o-", ms=3.5, lw=1.6, color=BLUE, label="whole genome")
    ax1.plot(hL_b[ok], h_b[ok], "s-", ms=3.5, lw=1.6, color=RED,
             label="background only (donor masked)")
    ax1.axhline(0.25, ls=":", lw=1.4, color=GREY)
    ax1.axhline(p_het, ls="--", lw=1.4, color=ORANGE)
    ax1.annotate("0.25  random null", xy=(21.5, 0.205), fontsize=7.5, color=GREY)
    ax1.annotate(f"{p_het:.3f}  genome composition", xy=(18.4, 0.283),
                 fontsize=7.5, color=ORANGE)
    ax1.annotate("the donor takes over\nthe surviving population", xy=(13.3, 0.62),
                 xytext=(14.8, 0.74), fontsize=7.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax1.annotate("repeats\n(Ty, paralogues)", xy=(18.1, 0.44), xytext=(19.6, 0.58),
                 fontsize=7.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax1.set_ylim(0, 1.08)
    ax1.set_xlim(k, 24)
    ax1.set_xlabel("L  (matched nt)")
    ax1.set_ylabel(r"$p(L) = P(M \geq L{+}1)\,/\,P(M \geq L)$")
    ax1.set_title(r"The model's $p$, measured", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, ls="--", alpha=0.35)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "spectrum.png")
    fig.savefig(out, dpi=170)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return out
