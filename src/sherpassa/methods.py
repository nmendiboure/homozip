"""
Aggregation and plotting utilities for the Gillespie trajectories.

Buckets are derived from the YAML `intermediates` list rather than
hardcoded, so the aggregation stays consistent with whatever bucket
discretization the model was generated with.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =====================================================================
# Bucket boundaries used for aggregation plots.
# Each tuple is (label, L_lo, L_hi) inclusive on both sides.
# Edit here if you want to change how complexes are grouped on plots;
# this is independent of the modelmaker discretization.
# =====================================================================
LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("8-9",   8,   9),
    ("12-21", 12,  21),
    ("24-30", 24,  30),
    ("45-95", 45,  95),
    ("140+",  140, 10**6),
)


# =====================================================================
# Aggregation
# =====================================================================

def make_group(
        result: np.ndarray,
        species_to_index: dict,
        intermediates: list[int],
) -> dict:
    """
    Convert a raw species-by-time matrix into a labeled dict of
    aggregated trajectories ready for plotting / saving.

    Parameters
    ----------
    result : (n_species, n_points) array
        Output of `simulation.run()`. Row 0 is time, row 1 is S, the
        rest follows the species declaration order.
    species_to_index : dict[str, int]
        Inverse mapping built in commands.py from `index_to_species`.
    intermediates : list[int]
        Length buckets used by the modelmaker.
    """

    n_pts = result.shape[1]

    def sum_species(names: list[str]) -> np.ndarray:
        """Row-sum a group of species; missing names are ignored."""
        idxs = [species_to_index[n] for n in names if n in species_to_index]
        if not idxs:
            return np.zeros(n_pts, dtype=np.float64)
        return result[idxs, :].sum(axis=0)

    # ------------------------------------------------------------------
    # Pull the canonical row indices
    # ------------------------------------------------------------------
    res: dict = {
        "time":       result[species_to_index["Time"], :],
        "free sites": result[species_to_index["S"], :],
        "Recombined": result[species_to_index["R"], :],
    }

    # ------------------------------------------------------------------
    # Bucketed totals (for the multi-panel trajectory plot)
    # ------------------------------------------------------------------
    for label, lo, hi in LENGTH_BUCKETS:
        in_bucket = [L for L in intermediates if lo <= L <= hi]
        hm_names  = [f"HM{L}" for L in in_bucket]
        ht_names  = [f"HT{L}" for L in in_bucket]
        homo   = sum_species(hm_names)
        hetero = sum_species(ht_names)
        res[f"homo {label} nts"]              = homo
        res[f"hetero {label} nts"]            = hetero
        res[f"all associations {label} nts"]  = homo + hetero

    # ------------------------------------------------------------------
    # D-loops — total and per-length (size-resolved)
    # ------------------------------------------------------------------
    dhm_names = [f"DHM{L}" for L in intermediates]
    dht_names = [f"DHT{L}" for L in intermediates]
    res["D-loop homologies"]   = sum_species(dhm_names)
    res["D-loop heterologies"] = sum_species(dht_names)

    for L in intermediates:
        res[f"DHM_L{L}"] = result[species_to_index[f"DHM{L}"], :]
        res[f"DHT_L{L}"] = result[species_to_index[f"DHT{L}"], :]

    # ------------------------------------------------------------------
    # DLC homologous: D-loop trajectory clipped at the first
    # recombination event. After recombination the DSB is "consumed"
    # so any remaining DHM signal would be a tracking artefact.
    # ------------------------------------------------------------------
    dlc = res["D-loop homologies"].copy()
    rec = res["Recombined"]
    if rec[-1] > 0:
        t_rec = int(np.argmax(rec > 0))
        dlc[t_rec + 1:] = 0
    res["DLC homologous"] = dlc

    # ------------------------------------------------------------------
    # Total occupancy (everything that's not free S)
    # ------------------------------------------------------------------
    all_complex_names = (
        [f"HM{L}"  for L in intermediates] +
        [f"HT{L}"  for L in intermediates] +
        dhm_names + dht_names
    )
    res["all"] = sum_species(all_complex_names) + res["Recombined"]

    return res


def aggregate_groups(groups: list[dict]) -> dict:
    """Element-wise mean across replicates."""
    n = len(groups)
    if n == 0:
        return {}

    acc: dict = {}
    for g in groups:
        for k, v in g.items():
            if k not in acc:
                acc[k] = v.astype(np.float64).copy()
            else:
                acc[k] += v
    return {k: v / n for k, v in acc.items()}


# =====================================================================
# Plotting
# =====================================================================

def plot_trajectories(one_res: dict, outpath: str = "", show: bool = False) -> None:
    """3x2 grid of trajectory panels."""

    t = one_res["time"]

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), constrained_layout=True)
    title_font  = {"fontsize": 12}
    label_font  = {"fontsize": 12}
    legend_font = fm.FontProperties(size=8)

    # ---- Panel 0: free vs occupied ----
    ax0 = axs[0, 0]
    ax0.plot(t, one_res["free sites"], color="blue", label="S (free sites)")
    ax0.plot(t, one_res["all"],        color="red",  label="Total occupancy")
    ax0.set_title("Filament binding dynamics", **title_font)
    ax0.set_xlabel("T", **label_font)
    ax0.set_ylabel("N", **label_font)
    ax0.legend(prop=legend_font)
    ax0.grid(True, linestyle="--", alpha=0.6)

    # ---- Panel 1: homologous complexes per bucket ----
    ax1 = axs[0, 1]
    ax1.set_title("Homologous pre-synaptic complexes", **title_font)

    # ---- Panel 2: heterologous complexes per bucket ----
    ax2 = axs[1, 0]
    ax2.set_title("Heterologous pre-synaptic complexes", **title_font)

    # ---- Panel 3: total complexes per bucket ----
    ax3 = axs[1, 1]
    ax3.set_title("All pre-synaptic complexes", **title_font)

    # ---- Panel 4: D-loops ----
    ax4 = axs[2, 0]
    ax4.set_title("D-loops", **title_font)

    # ---- Panel 5: recombined ----
    ax5 = axs[2, 1]
    ax5.set_title("Recombined state", **title_font)

    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_xlabel("T", **label_font)
        ax.set_ylabel("N", **label_font)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Route series to their panels by key prefix
    for k, v in one_res.items():
        if k.startswith("homo "):
            ax1.plot(t, v, label=k)
        elif k.startswith("hetero "):
            ax2.plot(t, v, label=k)
        elif k.startswith("all associations"):
            ax3.plot(t, v, label=k)
        elif k.startswith("D-loop"):
            ax4.plot(t, v, label=k)
        elif k.startswith("Recombined"):
            ax5.plot(t, v, label=k)

    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.legend(prop=legend_font)

    if show:
        plt.show()
    if outpath:
        fig.savefig(outpath, format="png")
    plt.close(fig)


def plot_dlc(aggr_dlc: np.ndarray, convo: int = 100, outpath: str = "", show: bool = False) -> None:
    """Plot the aggregated DLC homologous trace, optionally smoothed."""

    t = aggr_dlc.shape[0]
    if convo > 0:
        aggr_dlc = np.convolve(aggr_dlc, np.ones(convo) / convo, mode="same")

    xt = np.arange(t)
    fig, ax = plt.subplots()
    ax.plot(xt, aggr_dlc)
    ax.set_xlabel("T")
    ax.set_ylabel("DLC homologous (avg)")
    ax.grid(True, linestyle="--", alpha=0.6)

    if show:
        plt.show()
    if outpath:
        fig.savefig(outpath, format="png")
    plt.close(fig)
