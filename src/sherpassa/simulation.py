"""Single-cell Gillespie simulation driver."""

import numpy as np
import tellurium as te
from mpi4py import MPI
from numpy import random

from sherpassa import methods


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def run(
        uid: int,
        model_id: int,
        species_to_index: dict,
        my_model: str,
        params: dict,
        outdir: str,
) -> None:
    """
    Run one Gillespie replicate and dump its trajectory to disk.

    The simulation is split into two phases:

      1. Resection delay (Gamma-distributed)
            No chemistry happens. We do *not* call the solver: we just
            emit a flat segment with S = N during the delay window.
            This avoids the cost of a Tellurium run on a fully quiescent
            system, which used to dominate wall-time on short simulations.

      2. Synthesis
            Full Gillespie integration on the Antimony model from
            t = delay to t = n_timepoints, starting from S = N.
    """

    print(f"[Process {RANK}] :: SIMULATION {uid}", flush=True)

    # =================================================================
    # Per-cell randomness
    # =================================================================
    seed0 = params["seed_zero"]
    seed = seed0 + uid + model_id
    random.seed(seed)

    # =================================================================
    # Time grid
    # =================================================================
    n_timepoints: int = params["timepoints"]
    every: int        = params["every"]
    n_points          = n_timepoints // every

    n_species = len(species_to_index)

    # =================================================================
    # Resection delay (no chemistry)
    # =================================================================
    k     = params["gamma_k"]
    theta = params["gamma_theta"]
    delay = int(random.gamma(k, theta))
    delay = min(delay, n_timepoints - 1)

    n_pts_delay = max(0, int(round(delay / every)))
    n_pts_dyn   = n_points - n_pts_delay
    if n_pts_dyn <= 0:
        n_pts_delay = n_points - 1
        n_pts_dyn   = 1

    # =================================================================
    # Synthesis phase
    # =================================================================
    r = te.loada(my_model)
    r.integrator = "gillespie"
    r.integrator.seed = seed
    r.reset()

    t_start = n_pts_delay * every
    t_end   = n_timepoints

    s_dyn = r.simulate(t_start, t_end, n_pts_dyn)

    # =================================================================
    # Assemble the full trajectory
    # =================================================================
    # Convention from the previous pipeline: during the resection delay
    # we record S = N (filament available, nothing bound), and every
    # other species at 0. Tellurium's column ordering matches the
    # species declaration order in the model, which is also the order
    # we used to build species_to_index.
    s_total = np.zeros((n_species, n_points), dtype=np.float64)
    s_total[0, :] = np.arange(0, n_timepoints, every)
    s_total[1, :n_pts_delay] = params["N"]

    # s_dyn columns: [time, sp_1, sp_2, ...] in declaration order
    sl = slice(n_pts_delay, n_pts_delay + n_pts_dyn)
    for i in range(1, n_species):
        s_total[i, sl] = s_dyn[:, i]

    # =================================================================
    # Dump
    # =================================================================
    result_group = methods.make_group(
        s_total,
        species_to_index,
        params["intermediates"],
    )

    np.savez_compressed(f"{outdir}/simulation_{uid}.npz", **result_group)
