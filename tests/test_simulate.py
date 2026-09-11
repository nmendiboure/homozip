import math

import numpy as np
import pytest

pytest.importorskip("tellurium")

from homozip.model import Params      # noqa: E402
from homozip import theory, simulate  # noqa: E402


@pytest.mark.parametrize("m,delta", [(0, 0.05), (2, 0.15)])
def test_lone_seeds_draw_the_commit_probability(m, delta):
    prm = Params(max_mismatches=m, p_hom=1.0 - delta)
    n_rep = 300
    mean, se, _ = simulate.isolated_seed_commit_fraction(prm, n_rep, seed=7)
    expected = theory.commit_probability(prm, "H")
    tolerance = 4.0 * max(se, math.sqrt(expected * (1 - expected) / n_rep))
    assert abs(mean - expected) < tolerance


def test_ensemble_follows_the_exact_cdf(prm):
    n_cells = 60
    ens = simulate.run_ensemble(prm, n_cells, seed=11)
    t = ens["t"]
    dist = theory.search_time_distribution(prm, t)
    empirical = simulate.empirical_cdf(ens["first_commit"], t)
    for fraction in (0.5, 1.0):
        i = int(fraction * (t.size - 1))
        p = dist["cdf"][i]
        assert abs(empirical[i] - p) < 4.0 * math.sqrt(p * (1 - p) / n_cells) + 1e-9
    assert ens["S_mean"].mean() == pytest.approx(theory.free_sites(prm), rel=0.02)
    assert np.all(np.diff(ens["RH_mean"]) >= 0.0)
