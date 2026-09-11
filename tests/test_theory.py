import math

import numpy as np
import pytest

from homozip.model import Params
from homozip import theory


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.0, 0.02, 0.1])
def test_commit_probability_matches_the_binomial_form(m, delta):
    prm = Params(max_mismatches=m, p_hom=1.0 - delta)
    assert theory.commit_probability(prm, "H") == pytest.approx(
        theory.commit_probability_closed_form(prm, 1.0 - delta), rel=1e-12)


def test_no_tolerance_gives_a_power_law():
    prm = Params(p_hom=0.9)
    assert theory.commit_probability(prm) == pytest.approx(
        0.9 ** prm.n_steps * theory.kinetic_yield(prm), rel=1e-12)


@pytest.mark.parametrize("m", [0, 2])
def test_kinetics_cancel_out_of_the_discrimination(m):
    slow = Params(max_mismatches=m, koff0=0.7)
    fast = Params(max_mismatches=m, koff0=7.0, kext=50.0, lam=3.0)
    assert theory.discrimination(slow, 1.0, 0.271) == pytest.approx(
        theory.discrimination(fast, 1.0, 0.271), rel=1e-10)
    if m == 0:
        assert theory.discrimination(slow, 1.0, 0.271) == pytest.approx(
            (1.0 / 0.271) ** slow.n_steps, rel=1e-10)


def test_transit_time_does_not_depend_on_p():
    matched, divergent = Params(p_hom=1.0), Params(p_hom=0.7)
    assert theory.transit_time_mean(matched) == theory.transit_time_mean(divergent)
    t = np.linspace(0, 1, 50)
    assert np.allclose(theory.transit_time_pdf(matched, t),
                       theory.transit_time_pdf(divergent, t))
    assert np.trapz(t * theory.transit_time_pdf(matched, t), t) == pytest.approx(
        theory.transit_time_mean(matched), rel=1e-3)

    # Same for the length-dependent koff, where the density is phase-type.
    ladder = Params(lam=8.41)
    t = np.linspace(0, 1, 2000)
    assert np.trapz(t * theory.transit_time_pdf(ladder, t), t) == pytest.approx(
        theory.transit_time_mean(ladder), rel=1e-3)


def test_search_time_split_adds_up_to_the_commit_rate():
    prm = Params()
    split = theory.search_time_decomposition(prm)
    assert split["parallel"] == pytest.approx(1.0 / theory.commit_rate(prm), rel=1e-12)
    assert split["tau_diff"] > split["tau_off"] > split["tau_target"] > 0


def test_fidelity_bound_agrees_with_a_scan():
    prm = Params()
    for eps in (1e-2, 1e-3, 1e-4):
        n_min = theory.min_commitment_steps(prm, eps)
        assert theory.min_commitment_length(prm, eps) == prm.k_seed + math.ceil(n_min)
    assert 8 < theory.min_commitment_steps(prm, 1e-2) < 13
    assert theory.min_commitment_steps(prm, 1e-4) > theory.min_commitment_steps(prm, 1e-2)


def test_specificity_is_the_complement_of_the_false_fraction():
    prm = Params()
    odds = theory.false_commitment_odds(prm)
    assert theory.specificity(prm) == pytest.approx(1.0 / (1.0 + odds))
    assert odds < 1e-8


def test_best_koff0_estimate_is_close_to_the_scan():
    prm = Params()
    approx = theory.optimal_koff0_approx(prm)
    numeric, _ = theory.optimal_koff0_numeric(prm, n_points=600)
    assert numeric == pytest.approx(approx, rel=0.25)
    assert 0.1 < approx < 1.0


def test_first_passage_from_a_seed_reproduces_the_commit_probability():
    prm = Params(kon=0.0, p_hom=0.95, max_mismatches=1)
    t = np.linspace(0.0, 60.0 / prm.koff0, 400)
    fp = theory.first_passage(prm, t, start="H8")
    assert fp["cdf_H"][-1] == pytest.approx(theory.commit_probability(prm, "H"), rel=1e-8)
    assert fp["cdf_X"][-1] == 0.0
    assert np.all(np.diff(fp["cdf_H"]) >= -1e-15)


def test_the_hazard_settles_at_the_steady_state_rate():
    prm = Params()
    t = np.linspace(0.0, 40.0, 2001)
    fp = theory.first_passage(prm, t)
    assert fp["hazard_H"][0] == 0.0
    assert prm.n_sites * fp["hazard_H"][-1] == pytest.approx(
        theory.commit_rate(prm), rel=2e-3)


def test_search_time_distribution_and_its_mean():
    prm = Params()
    t = np.linspace(0.0, 4000.0, 4001)
    dist = theory.search_time_distribution(prm, t)
    assert dist["cdf"][-1] > 0.99
    assert np.trapz(dist["pdf"], t) == pytest.approx(dist["cdf"][-1], rel=1e-3)
    assert theory.mean_search_time_exact(prm) == pytest.approx(
        theory.mean_search_time_qss(prm), rel=1e-2)


def test_a_background_seed_holds_its_site_for_one_koff():
    prm = Params()
    tau_background = theory.occupancy_time(prm, "X")
    assert tau_background == pytest.approx(1.0 / prm.koff0, rel=0.05)
    assert theory.occupancy_time(prm, "H") < tau_background
