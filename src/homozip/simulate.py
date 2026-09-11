"""
Gillespie runs, with tellurium.

Two experiments, each with an exact counterpart in theory.py:

    isolated_seed_commit_fraction   one seed, nucleation off, so the run is
                                    a Bernoulli draw of commit_probability
    run_ensemble                    the full N-site model, giving the time
                                    of the first donor commitment per cell
"""

from __future__ import annotations

import numpy as np
import tellurium as te

from .model import Params, build_network, to_antimony, state_name


def load(prm: Params):
    """A roadrunner instance with the Gillespie integrator armed."""
    rr = te.loada(to_antimony(build_network(prm), prm))
    rr.integrator = "gillespie"
    rr.integrator.variable_step_size = False
    return rr


def isolated_seed_commit_fraction(prm: Params, n_rep: int, seed: int | None = None,
                                  track: str = "H") -> tuple[float, float, int]:
    """Fraction of lone seeds that reach L_commit: (mean, standard error, hits).

    Each replicate starts with one joint at the seed length, no free site
    and kon = 0, and runs long enough for the slowest blocked joint to have
    fallen off."""
    rr = load(prm)
    seed = prm.seed if seed is None else seed
    start = state_name(track, prm.k_seed)
    target = "RH" if track == "H" else "RX"
    rr.timeCourseSelections = ["time", target]
    t_end = 40.0 / min(prm.koff(l) for l in prm.lengths)

    hits = 0
    for i in range(n_rep):
        rr.reset()
        rr.integrator.seed = seed + i
        rr["kon"] = 0.0
        rr["S"] = 0
        rr[start] = 1
        hits += int(round(rr.simulate(0.0, t_end, 2)[-1, 1]))

    q = hits / n_rep
    return q, float(np.sqrt(max(q * (1.0 - q), 1e-12) / n_rep)), hits


def run_ensemble(prm: Params, n_cells: int | None = None, seed: int | None = None,
                 t_end: float | None = None, n_points: int | None = None,
                 progress: bool = False) -> dict:
    """The full model, n_cells independent runs on a fixed output grid.

    first_commit holds the first grid point at which RH > 0, so the true
    time lies in the preceding interval, and NaN for a cell that never
    commits. That censoring says something, so it is not dropped here."""
    n_cells = prm.n_cells if n_cells is None else n_cells
    seed = prm.seed if seed is None else seed
    t_end = prm.t_end if t_end is None else t_end
    n_points = prm.n_points if n_points is None else n_points

    rr = load(prm)
    rr.timeCourseSelections = ["time", "S", "RH", "RX"]
    times = np.full(n_cells, np.nan)
    acc = np.zeros((n_points, 3))
    grid = None

    for i in range(n_cells):
        rr.reset()
        rr.integrator.seed = seed + i
        out = np.asarray(rr.simulate(0.0, t_end, n_points))
        grid = out[:, 0]
        acc += out[:, 1:4]
        if out[-1, 2] > 0:
            times[i] = out[int(np.argmax(out[:, 2] > 0)), 0]
        if progress and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_cells} cells", flush=True)

    acc /= n_cells
    return {"t": grid, "first_commit": times,
            "S_mean": acc[:, 0], "RH_mean": acc[:, 1], "RX_mean": acc[:, 2]}


def empirical_cdf(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P[T <= t] from first-passage times, counting the NaNs as censored."""
    obs = times[~np.isnan(times)]
    return np.array([(obs <= ti).sum() for ti in t], dtype=float) / times.size
