import numpy as np
import pytest

from homozip.model import Params, build_network, to_antimony, export_sbml
from homozip import theory


@pytest.mark.parametrize("m", [0, 1, 2])
def test_species_and_reaction_counts(m):
    prm = Params(max_mismatches=m)
    net = build_network(prm)
    n = prm.n_steps
    # S, RH, RX, then per track: (m+1) zipping ladders and one blocked ladder.
    assert len(net.species) == 3 + 2 * n * (m + 2)
    # Two nucleations, then per track and per ladder: zip, mismatch, fall off,
    # plus one fall-off per blocked state.
    assert len(net.reactions) == 2 + 2 * n * (3 * m + 4)
    assert len(set(net.species)) == len(net.species)
    assert net.species[0] == "S" and net.species[-2:] == ["RH", "RX"]


def test_every_reaction_names_a_declared_species():
    net = build_network(Params(max_mismatches=1))
    declared = set(net.species)
    for r in net.reactions:
        assert r.src in declared and r.dst in declared
    assert not any(r.src in net.absorbing for r in net.reactions)


def test_generator_rows_sum_to_zero():
    net = build_network(Params(max_mismatches=1, lam=8.41))
    q, idx = theory.generator(net)
    assert np.allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diag(q) <= 0.0)
    for name in net.absorbing:
        assert np.all(q[idx[name]] == 0.0)


def test_antimony_loads_and_exports_sbml():
    te = pytest.importorskip("tellurium")
    prm = Params(max_mismatches=1)
    net = build_network(prm)
    src = to_antimony(net, prm)
    rr = te.loada(src)
    assert set(rr.getFloatingSpeciesIds()) == set(net.species)
    assert len(rr.getReactionIds()) == len(net.reactions)
    assert rr["S"] == prm.n_sites
    assert "<sbml" in export_sbml(src)


def test_scalar_p_stays_tunable_but_a_profile_is_written_out():
    prm = Params()
    assert "p_het = 0.271" in to_antimony(build_network(prm), prm)

    prm = Params(p_het=tuple([0.3] * 22))
    src = to_antimony(build_network(prm), prm)
    assert "p_het = " not in src
    assert "kext * 0.3" in src


def test_from_config_aliases_and_validation():
    prm = Params.from_config({"N": 10, "L_commit": 20, "delta": 0.05,
                              "seed_zero": 3, "lam": 0})
    assert (prm.n_sites, prm.l_commit, prm.seed, prm.lam) == (10, 20, 3, None)
    assert prm.p_hom == pytest.approx(0.95)

    with pytest.raises(ValueError):
        Params.from_config({"delta": 0.1, "p_hom": 0.9})
    with pytest.raises(ValueError):
        Params.from_config({"nope": 1})
    with pytest.raises(ValueError):
        Params(l_commit=8)
    with pytest.raises(ValueError):
        Params(p_het=(0.2, 0.3))


def test_uid_ignores_simulation_settings():
    assert Params(n_cells=10).uid() == Params(n_cells=99, seed=5).uid()
    assert Params().uid() != Params(koff0=0.71).uid()
