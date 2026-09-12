"""
One PDF per panel, all from the exact solution. Nothing is simulated.

    divergence.pdf          P_commit against donor divergence, for m = 0, 1, 2
    fidelity.pdf            false commitments per true one against L_commit
    commitment_length.pdf   mean search time against L_commit, three divergences
    association.pdf         commitment rate against kon: saturation
    occupancy.pdf           share of busy sites against kon: concomitant joints
    stability.pdf           commitment rate against koff, three loads
    first_passage.pdf       cells with a donor commitment against time
    hazard.pdf              the hazard: one transit and one blocked joint are the memory
    spectrum_survival.pdf   the match-length spectrum behind f
    spectrum_hazard.pdf     the measured p(L) behind p_het

Each panel is about 60 x 52 mm with 7.5 pt text, cropped to its content,
so that it drops into a multi-panel figure without rescaling.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from .model import Params            # noqa: E402
from . import model, genome            # noqa: E402

# Ordered levels of one parameter share one hue, light to dark. Orange
# marks the reference value of params.yaml. Grey is for guides.
RAMP = ("#79ade6", "#2a78d6", "#0f3d7a")
ACCENT, MUTED, GREY = "#eb6834", "#52514e", "#9a9a94"
PANEL = (60 / 25.4, 52 / 25.4)
STYLE = {
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
}


def _scalar(prm: Params) -> Params:
    # Panels that sweep L_commit cannot carry a profile of fixed length, so
    # a measured p(L) is replaced by its mean there.
    changes = {key: float(np.mean(getattr(prm, key)))
               for key in ("p_hom", "p_het")
               if not isinstance(getattr(prm, key), float)}
    return prm.with_(**changes) if changes else prm


def _end_label(ax, x, y, text, color, dx=3, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=6.5, va="center")


def _note(ax, x, y, text, color=MUTED, **kw):
    ax.text(x, y, text, color=color, fontsize=6, **kw)


# =====================================================================
# Sequence and length
# =====================================================================

def divergence(ax, prm: Params) -> None:
    """The divergence law, and the shoulder a tolerance adds."""
    prm = _scalar(prm)
    deltas = np.linspace(0.0, 0.20, 81)
    for m, c in zip((0, 1, 2), RAMP):
        y = [model.commit_probability(prm.with_(max_mismatches=m, p_hom=1 - d))
             for d in deltas]
        ax.semilogy(deltas, y, color=c)
        _end_label(ax, deltas[-1], y[-1], f"$m={m}$", c)
    _note(ax, 0.03, 0.06, f"slope $= n = L_c - k = {prm.n_steps}$",
          transform=ax.transAxes)
    ax.set_ylim(1e-3, 2)
    ax.set_xlim(0, 0.235)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel(r"donor divergence $\delta$")
    ax.set_ylabel(r"$P_{\mathrm{commit}}$ per seed")
    ax.set_title("Divergence and tolerance", loc="left")


def fidelity(ax, prm: Params, lcs=np.arange(10, 61)) -> None:
    """False commitments per true one against the test length: the
    fidelity bound, and how little length three more decades cost."""
    prm = _scalar(prm)
    odds = np.array([model.false_commitment_odds(prm.with_(l_commit=int(l)))
                     for l in lcs])
    ax.semilogy(lcs, odds, color=RAMP[1])
    marks = []
    for eps in (1e-2, 1e-3, 1e-4):
        lmin = int(np.ceil(prm.k_seed + model.min_commitment_steps(prm, eps)))
        marks.append(lmin)
        ax.axhline(eps, color=GREY, lw=0.6, ls=":")
        ax.plot(lmin, odds[int(lmin - lcs[0])], "o", ms=3.5, color=RAMP[1], zorder=5)
    _note(ax, lcs[-1] + 1, 3e-2,
          "tolerated false rate\n$10^{-2}$, $10^{-3}$, $10^{-4}$", ha="right",
          va="bottom")
    ax.annotate(", ".join(map(str, marks)) + " nt",
                (marks[-1], odds[int(marks[-1] - lcs[0])]), xytext=(6, -10),
                textcoords="offset points", color=MUTED, fontsize=6)
    ax.axvline(prm.l_commit, color=GREY, lw=0.6, ls=":")
    _note(ax, prm.l_commit + 0.7, 1e-24, f"{prm.l_commit} nt\nreference", va="bottom")
    ax.text(0.05, 0.06, r"$\frac{1-f}{f\,c}\left(\frac{p_{het}}{p_{hom}}\right)^{n}$",
            transform=ax.transAxes, color=RAMP[1], fontsize=8)
    ax.set_xlim(lcs[0], lcs[-1] + 2)
    ax.set_ylim(1e-26, 1e4)
    ax.set_yticks([1e0, 1e-6, 1e-12, 1e-18, 1e-24])
    ax.set_xlabel(r"commitment length $L_c$ (nt)")
    ax.set_ylabel("false / true commitments")
    ax.set_title("Fidelity from length", loc="left")


def commitment_length(ax, prm: Params, lcs=np.arange(10, 61)) -> None:
    """What a longer test costs: nothing on the right donor, a factor
    exp(n delta) on a divergent one."""
    prm = _scalar(prm)
    lo, hi = np.inf, 0.0
    for d, c in zip((0.0, 0.02, 0.05), RAMP):
        t = [model.mean_search_time(prm.with_(l_commit=int(l), p_hom=1 - d))
             for l in lcs]
        ax.semilogy(lcs, t, color=c)
        _end_label(ax, lcs[-1], t[-1], rf"$\delta={d:g}$", c)
        lo, hi = min(lo, min(t)), max(hi, max(t))
    lmin = int(np.ceil(prm.k_seed + model.min_commitment_steps(prm, 1e-3)))
    ax.axvspan(lcs[0], lmin, color=GREY, alpha=0.12, lw=0)
    _note(ax, lmin + 0.7, hi * 1.5, "too short:\nmore than one\nfalse per thousand",
          va="top")
    ax.axvline(prm.l_commit, color=GREY, lw=0.6, ls=":")
    ax.set_xlim(lcs[0], lcs[-1] + 14)
    ax.set_ylim(lo / 1.5, hi * 2)
    ax.set_xlabel(r"commitment length $L_c$ (nt)")
    ax.set_ylabel("mean search time (min)")
    ax.set_title("Cost of length", loc="left")


# =====================================================================
# Rates
# =====================================================================

def _kon_grid():
    return np.logspace(-3, 1, 120)


def _mark_reference(ax, x: float, y_text: float, ha="left") -> None:
    ax.axvline(x, color=ACCENT, lw=0.8)
    _note(ax, x * (1.2 if ha == "left" else 0.8), y_text, "reference", color=ACCENT,
          va="top", ha=ha)


def association(ax, prm: Params) -> None:
    """The commitment rate against kon saturates at a plateau set by koff,
    because blocked background joints hold the sites."""
    kons = _kon_grid()
    for ko, c in zip((0.07, 0.7, 7.0), RAMP):
        rate = [model.commit_rate(prm.with_(kon=k, koff0=ko)) for k in kons]
        ax.loglog(kons, rate, color=c, label=rf"$k_{{\mathrm{{off}}}}={ko:g}$ min$^{{-1}}$")
    # The rate if every site stayed free, kon f N P_commit: what the sites
    # held by blocked background joints take away.
    lin = prm.f * prm.contact * prm.n_sites * model.commit_probability(prm, "H") * kons
    ax.loglog(kons, lin, color=GREY, lw=0.7, ls="--")
    top = float(lin[-1]) * 2
    _note(ax, 0.12, float(np.interp(0.12, kons, lin)) * 1.4, "every site free",
          color=GREY, rotation=38, rotation_mode="anchor", va="bottom")
    _mark_reference(ax, prm.kon, top * 0.7)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5)
    ax.set_xlim(1e-3, 3e1)
    ax.set_ylim(top * 1e-5, top)
    ax.set_xlabel(r"$k_{\mathrm{on}}$ per site (min$^{-1}$)")
    ax.set_ylabel(r"commitment rate (min$^{-1}$)")
    ax.set_title("Association load", loc="left")


def occupancy(ax, prm: Params) -> None:
    """How many of the N sites hold a joint at once. This is the
    concomitance of the search: at the reference kon the filament is
    empty; at kon = 1 per site more than half of it is busy, which is the
    intersegmental search of Forget and Kowalczykowski (2012)."""
    kons = _kon_grid()
    for ko, c in zip((0.07, 0.7, 7.0), RAMP):
        busy = [model.occupied_fraction(prm.with_(kon=k, koff0=ko)) for k in kons]
        ax.semilogx(kons, busy, color=c, label=rf"$k_{{\mathrm{{off}}}}={ko:g}$ min$^{{-1}}$")
    _mark_reference(ax, prm.kon, 0.36)
    ax.legend(loc="center left", frameon=False, handlelength=1.5,
              bbox_to_anchor=(0.0, 0.62))
    ax.set_xlim(1e-3, 3e1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$k_{\mathrm{on}}$ per site (min$^{-1}$)")
    ax.set_ylabel(f"busy sites, share of $N = {prm.n_sites}$")
    ax.set_title("Concomitant joints", loc="left")


def stability(ax, prm: Params) -> None:
    """Filament stability has an interior optimum, broad on an empty
    filament and sharp on a crowded one. Only koff / kext enters."""
    grid = prm.kext * np.logspace(-5, 0.5, 250)
    for mult, c in zip((0.01, 0.1, 1), RAMP):
        p = prm.with_(kon=prm.kon * mult)
        rate = np.array([model.commit_rate(p.with_(koff0=k)) for k in grid])
        rate /= rate.max()
        ax.loglog(grid, rate, color=c, label=f"{p.kon:g}")
        i = int(np.argmax(rate))
        ax.plot(grid[i], 1.0, "o", ms=3.5, color=c, zorder=5)
    _mark_reference(ax, prm.koff0, 3e-3, ha="right")
    # The legend sits above the axes: the curves fill the plot.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              handlelength=1.2, ncol=4, columnspacing=0.8, handletextpad=0.4,
              borderaxespad=0.0)
    ax.plot([], [], " ", label=r"$k_{\mathrm{on}}$ (min$^{-1}$):")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              handlelength=1.2, ncol=4, columnspacing=0.8, handletextpad=0.4,
              borderaxespad=0.0)
    ax.set_xlim(3e-3, 1e3)
    ax.set_ylim(1e-3, 2)
    ax.set_xlabel(r"$k_{\mathrm{off}}$ (min$^{-1}$),  $k_{\mathrm{ext}} = "
                  rf"{prm.kext:g}$ nt min$^{{-1}}$")
    ax.set_ylabel("commitment rate, relative to best")
    ax.set_title("Filament stability", loc="left", pad=14)


# =====================================================================
# Time
# =====================================================================

def first_passage(ax, prm: Params, t_end: float = 480.0) -> None:
    """Fraction of cells with a donor commitment against time, for three
    association rates: exponential from t = 0 whatever the rate. Dots,
    the half time of each curve."""
    t = np.linspace(0.0, t_end, 481)
    for mult, c in zip((0.1, 1, 10), RAMP):
        p = prm.with_(kon=prm.kon * mult)
        cdf = model.search_time_distribution(p, t)["cdf"]
        ax.plot(t / 60, cdf, color=c, label=rf"$k_{{\mathrm{{on}}}}={p.kon:g}$ min$^{{-1}}$")
        if cdf[-1] > 0.5:
            ax.plot(np.interp(0.5, cdf, t) / 60, 0.5, "o", ms=3.5, color=c, zorder=5)
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    _note(ax, t_end / 60 * 1.02, 0.5, r"$t_{1/2}$", ha="right", va="bottom")
    ax.legend(loc="lower right", frameon=False, handlelength=1.5)
    ax.set_xlim(0, t_end / 60 * 1.04)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, int(t_end / 60) + 1, 2))
    ax.set_xlabel("time after the search starts (h)")
    ax.set_ylabel("cells with a donor commitment")
    ax.set_title("First donor commitment", loc="left")


def hazard(ax, prm: Params) -> None:
    """The hazard of a donor commitment for a filament of N fresh sites,
    relative to its steady-state value. It rises over one transit, then
    settles over one background occupancy, 1 / koff, as the sites fill
    with blocked joints. Those two times are the whole memory of the
    search, which is why first_passage.pdf is exponential."""
    rate = model.commit_rate(prm)
    transit = model.transit_time_mean(prm)
    t_s = np.linspace(0.0, 60.0 * 5.0 / prm.koff0, 400)
    fp = model.first_passage(prm, t_s / 60.0)
    ax.plot(t_s, prm.n_sites * fp["hazard_H"] / rate, color=RAMP[1])
    ax.axhline(1.0, color=GREY, lw=0.7, ls="--")
    _note(ax, t_s[-1], 0.96, r"$\lambda = k_{on} f S_{free} P_{commit}$",
          color=GREY, ha="right", va="top")
    for x, text in ((60.0 * transit, f"one transit, {60 * transit:.0f} s"),
                    (60.0 / prm.koff0, f"one blocked joint, {60 / prm.koff0:.0f} s")):
        ax.axvline(x, color=GREY, lw=0.6, ls=":")
        _note(ax, x * 1.08, 0.12, text, rotation=90, va="bottom")
    ax.set_ylim(0, None)
    ax.set_xlim(0, t_s[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"hazard $/\,\lambda$")
    ax.set_title("The memory of the search", loc="left")


# =====================================================================
# The genome measurement
# =====================================================================

def spectrum_survival(ax, agg: dict, bg: dict, l_commit: int) -> None:
    k = agg["ksize"]
    L_a, S_a = np.array(agg["L"], float), np.array(agg["survival"])
    L_b, S_b = np.array(bg["L"], float), np.array(bg["survival"])
    f_donor = genome.donor_fraction(agg, l_commit)
    p_het = genome.background_p(bg)

    ax.semilogy(L_a, np.where(S_a > 0, S_a, np.nan), "o-", ms=2.5, lw=1.2,
                color=RAMP[2], label="whole genome")
    ax.semilogy(L_b, np.where(S_b > 0, S_b, np.nan), "s-", ms=2.5, lw=1.2,
                color=RAMP[0], label="donor masked")
    ax.semilogy(L_a, p_het ** (L_a - k), "--", lw=0.8, color=GREY,
                label=rf"${p_het:.3f}^{{\,L-k}}$")
    ax.semilogy(L_a, (1 - f_donor) * p_het ** (L_a - k) + f_donor, "-", lw=0.8,
                color=ACCENT, label=r"$(1-f)\,p_{het}^{L-k} + f$")
    _note(ax, 17, f_donor * 3.0, f"plateau: the donor alone\n$f = {f_donor:.2e}$",
          color=ACCENT)
    ax.axvline(l_commit, color=GREY, lw=0.6, ls=":")
    _note(ax, l_commit + 0.5, 3e-6, "$L_c$", va="bottom")
    ax.legend(loc="lower left", frameon=False, handlelength=1.5)
    ax.set_ylim(1e-7, 2)
    ax.set_xlim(k, 34)
    ax.set_xlabel("$L$, matched nt")
    ax.set_ylabel(r"$P(M \geq L)$ per (site, locus)")
    ax.set_title("Match-length spectrum", loc="left")


def spectrum_hazard(ax, agg: dict, bg: dict) -> None:
    k = agg["ksize"]
    p_het = genome.background_p(bg)
    hL_a, h_a = np.array(agg["hazard_L"], float), np.array(agg["hazard"])
    hL_b, h_b = np.array(bg["hazard_L"], float), np.array(bg["hazard"])
    ok = h_b > 0

    ax.plot(hL_a, h_a, "o-", ms=2.5, lw=1.2, color=RAMP[2], label="whole genome")
    ax.plot(hL_b[ok], h_b[ok], "s-", ms=2.5, lw=1.2, color=RAMP[0], label="donor masked")
    ax.axhline(0.25, ls=":", lw=0.8, color=GREY)
    ax.axhline(p_het, ls="--", lw=0.8, color=GREY)
    _note(ax, 23.8, 0.25, "0.25, random", color=GREY, ha="right", va="top")
    _note(ax, 23.8, p_het, f"{p_het:.3f}, composition", color=GREY, ha="right",
          va="bottom")
    ax.annotate("the donor takes over", xy=(13.3, 0.62), xytext=(14.5, 0.78),
                fontsize=6, color=RAMP[2],
                arrowprops=dict(arrowstyle="->", color=RAMP[2], lw=0.7))
    ax.annotate("repeats", xy=(18.1, 0.44), xytext=(19.5, 0.58), fontsize=6,
                color=RAMP[0], arrowprops=dict(arrowstyle="->", color=RAMP[0], lw=0.7))
    ax.legend(loc="upper left", frameon=False, handlelength=1.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(k, 24)
    ax.set_xlabel("$L$, matched nt")
    ax.set_ylabel(r"$p(L) = P(M \geq L{+}1)\,/\,P(M \geq L)$")
    ax.set_title("The model's $p$, measured", loc="left")


# =====================================================================
# Files
# =====================================================================

def _plot(outdir: str, name: str, draw) -> str:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=PANEL)
        draw(ax)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"{name}.pdf")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return out


def figure_model(prm: Params, outdir: str) -> list[str]:
    panels = [("divergence", divergence), ("fidelity", fidelity),
              ("commitment_length", commitment_length), ("association", association),
              ("occupancy", occupancy), ("stability", stability),
              ("first_passage", first_passage), ("hazard", hazard)]
    return [_plot(outdir, name, lambda ax, fn=fn: fn(ax, prm)) for name, fn in panels]


def figure_spectrum(genome_json: str, background_json: str, outdir: str,
                    l_commit: int = 30) -> list[str]:
    agg, bg = genome.load(genome_json), genome.load(background_json)
    return [_plot(outdir, "spectrum_survival",
                  lambda ax: spectrum_survival(ax, agg, bg, l_commit)),
            _plot(outdir, "spectrum_hazard", lambda ax: spectrum_hazard(ax, agg, bg))]
