import math

import numpy as np
import pytest

from homozip import model
from homozip.model import Params


# ---- the chain ------------------------------------------------------

@pytest.mark.parametrize("m", [0, 1, 2])
def test_state_and_transition_counts(m):
    prm = Params(max_mismatches=m)
    states, rx = model.build_chain(prm)
    n = prm.n_steps
    # S, RH, RX, then per track: (m+1) zipping ladders and one blocked ladder.
    assert len(states) == 3 + 2 * n * (m + 2)
    # Two nucleations, then per track and per ladder: zip, mismatch, fall off,
    # plus one fall-off per blocked state.
    assert len(rx) == 2 + 2 * n * (3 * m + 4)
    assert len(set(states)) == len(states)
    declared = set(states)
    assert all(r.src in declared and r.dst in declared for r in rx)
    assert not any(r.src in ("RH", "RX") for r in rx)


def test_generator_rows_sum_to_zero():
    q, idx = model.generator(Params(max_mismatches=1, lam=8.41))
    assert np.allclose(q.sum(axis=1), 0.0)
    assert np.all(np.diag(q) <= 0.0)
    assert np.all(q[idx["RH"]] == 0.0) and np.all(q[idx["RX"]] == 0.0)


def test_params_aliases_profiles_and_validation():
    prm = Params.from_config({"N": 10, "L_commit": 20, "delta": 0.05, "lam": 0})
    assert (prm.n_sites, prm.l_commit, prm.lam) == (10, 20, None)
    assert prm.p_hom == pytest.approx(0.95)

    profile = Params(p_het=tuple(0.2 + 0.01 * i for i in range(22)))
    assert profile.p("X", 8) == pytest.approx(0.20)
    assert profile.p("X", 29) == pytest.approx(0.41)

    for bad in ({"delta": 0.1, "p_hom": 0.9}, {"nope": 1}):
        with pytest.raises(ValueError):
            Params.from_config(bad)
    with pytest.raises(ValueError):
        Params(l_commit=8)
    with pytest.raises(ValueError):
        Params(p_het=(0.2, 0.3))


# ---- pen and paper --------------------------------------------------

@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.0, 0.02, 0.1])
def test_commit_probability_matches_the_binomial_form(m, delta):
    prm = Params(max_mismatches=m, p_hom=1.0 - delta)
    assert model.commit_probability(prm, "H") == pytest.approx(
        model.commit_probability_closed_form(prm, 1.0 - delta), rel=1e-12)
    if m == 0:
        assert model.commit_probability(prm) == pytest.approx(
            (1.0 - delta) ** prm.n_steps * model.kinetic_yield(prm), rel=1e-12)


@pytest.mark.parametrize("m", [0, 2])
def test_kinetics_cancel_out_of_the_discrimination(m):
    slow = Params(max_mismatches=m, koff0=0.7)
    fast = Params(max_mismatches=m, koff0=7.0, kext=50.0, lam=3.0)
    assert model.discrimination(slow, 1.0, 0.271) == pytest.approx(
        model.discrimination(fast, 1.0, 0.271), rel=1e-10)
    if m == 0:
        assert model.discrimination(slow, 1.0, 0.271) == pytest.approx(
            (1.0 / 0.271) ** slow.n_steps, rel=1e-10)


def test_transit_time_does_not_depend_on_p():
    assert model.transit_time_mean(Params(p_hom=1.0)) == \
        model.transit_time_mean(Params(p_hom=0.7))


def test_fidelity_bound_is_the_shortest_test_that_meets_epsilon():
    prm = Params()
    for eps in (1e-2, 1e-3, 1e-4):
        lc = prm.k_seed + math.ceil(model.min_commitment_steps(prm, eps))
        assert model.false_commitment_odds(prm.with_(l_commit=lc)) <= eps
        assert model.false_commitment_odds(prm.with_(l_commit=lc - 1)) > eps
    assert 8 < model.min_commitment_steps(prm, 1e-2) < 13


def test_contact_scales_only_the_donor():
    # With the donor always within reach the bound is a sequence statement.
    full = Params(contact=1.0)
    assert full.k_seed + math.ceil(model.min_commitment_steps(full, 1e-3)) == 19
    # A donor within reach a fraction c of the time costs ln(1/c)/ln(p_hom/p_het)
    # more checks, and divides the commitment rate by c on an empty filament.
    part = full.with_(contact=0.1, kon=1e-3)
    extra = math.log(10.0) / math.log(1.0 / part.p_het)
    assert model.min_commitment_steps(part, 1e-3) == pytest.approx(
        model.min_commitment_steps(full, 1e-3) + extra)
    assert model.commit_rate(part) == pytest.approx(
        0.1 * model.commit_rate(full.with_(kon=1e-3)), rel=1e-3)
    assert model.free_sites(part) == pytest.approx(model.free_sites(full.with_(kon=1e-3)),
                                                   rel=1e-3)


def test_half_time_reads_the_exact_distribution():
    prm = Params()
    t_half = model.half_time(prm)
    t = np.linspace(0.0, 2.0 * t_half, 2001)
    assert np.interp(t_half, t, model.search_time_distribution(prm, t)["cdf"]) \
        == pytest.approx(0.5, abs=2e-3)
    assert t_half < math.log(2.0) * model.mean_search_time(prm)


def test_a_background_seed_holds_its_site_for_one_koff():
    prm = Params()
    tau_x = model.occupancy_time(prm, "X")
    assert tau_x == pytest.approx(1.0 / prm.koff0, rel=0.05)
    assert model.occupancy_time(prm, "H") < tau_x


def test_occupancy_and_search_time_add_up():
    prm = Params(kon=1.0)
    assert model.free_sites(prm) + prm.n_sites * model.occupied_fraction(prm) \
        == pytest.approx(prm.n_sites)
    assert model.free_sites(Params(kon=0.01)) > model.free_sites(prm)
    split = model.search_time_decomposition(prm)
    assert split["parallel"] == pytest.approx(1.0 / model.commit_rate(prm), rel=1e-12)
    # Zipping the donor itself is never where the time goes.
    assert min(split["tau_diff"], split["tau_off"]) > 1e3 * split["tau_target"] > 0


def test_best_koff0_is_where_the_scan_peaks():
    prm = Params()
    grid = prm.kext * np.logspace(-5, 1, 600)
    rates = [model.commit_rate(prm.with_(koff0=k)) for k in grid]
    assert grid[int(np.argmax(rates))] == pytest.approx(model.optimal_koff0(prm), rel=0.25)


# ---- exact first passage --------------------------------------------

def test_first_passage_from_a_seed_reproduces_the_commit_probability():
    prm = Params(kon=0.0, p_hom=0.95, max_mismatches=1)
    t = np.linspace(0.0, 60.0 / prm.koff0, 400)
    fp = model.first_passage(prm, t, start="H8")
    assert fp["cdf_H"][-1] == pytest.approx(model.commit_probability(prm, "H"), rel=1e-8)
    assert fp["cdf_X"][-1] == 0.0
    assert np.all(np.diff(fp["cdf_H"]) >= -1e-15)


def test_the_hazard_settles_at_the_steady_state_rate():
    prm = Params()
    fp = model.first_passage(prm, np.linspace(0.0, 40.0, 2001))
    assert fp["hazard_H"][0] == 0.0
    assert prm.n_sites * fp["hazard_H"][-1] == pytest.approx(
        model.commit_rate(prm), rel=2e-3)


def test_search_time_distribution_and_its_mean():
    prm = Params()
    t = np.linspace(0.0, 6.0 * model.mean_search_time(prm), 4001)
    dist = model.search_time_distribution(prm, t)
    assert dist["cdf"][-1] > 0.99
    assert np.trapz(dist["pdf"], t) == pytest.approx(dist["cdf"][-1], rel=5e-3)
    # A fresh filament has every site free, so it starts faster than the
    # steady state: the exact mean is below the estimate, by about one
    # background occupancy. On an empty filament the two agree.
    assert model.mean_search_time_exact(prm) < model.mean_search_time(prm)
    dilute = Params(kon=0.01)
    assert model.mean_search_time_exact(dilute) == pytest.approx(
        model.mean_search_time(dilute), rel=1e-2)
