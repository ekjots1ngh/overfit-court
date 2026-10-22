"""Random matrix theory distinct bets, and the Reality Check."""

import numpy as np

from overfit_court import rmt, stats
from overfit_court.data import CALM_VOL, TURBULENT_VOL


def test_distinct_bets_counts_independent_groups():
    rng = np.random.default_rng(0)
    T = 1500
    independent = rng.normal(0, 0.01, (T, 40))
    assert 39 < rmt.spectrum(independent).distinct_bets <= 40

    five_bets = rng.normal(0, 0.01, (T, 5))
    eight_copies_each = np.repeat(five_bets, 8, axis=1) + rng.normal(0, 0.002, (T, 40))
    assert 4.5 < rmt.spectrum(eight_copies_each).distinct_bets < 6

    duplicated = np.hstack([independent, independent])
    assert abs(rmt.spectrum(duplicated).distinct_bets - rmt.spectrum(independent).distinct_bets) < 1


def test_distinct_bets_is_unbiased_with_many_strategies_and_few_days():
    """Removing sampling noise pair by pair used to undercount badly here (287 of 400)."""
    noise = np.random.default_rng(1).normal(0, 0.01, (500, 400))
    assert rmt.spectrum(noise).distinct_bets > 390


def test_shared_volatility_regimes_are_not_mistaken_for_shared_bets():
    """Unrelated strategies on one market share its calm and turbulent spells."""
    rng = np.random.default_rng(2)
    state = np.zeros(1890, dtype=int)
    for t in range(1, len(state)):
        stay = 0.99 if state[t - 1] == 0 else 0.97
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    vol = np.where(state == 1, TURBULENT_VOL, CALM_VOL)
    unrelated = rng.normal(0, 1, (len(vol), 200)) * vol[:, None]
    spectrum = rmt.spectrum(unrelated)
    assert spectrum.distinct_bets > 195
    assert spectrum.noise_edge > spectrum.mp_upper  # shared volatility widens the noise band
    assert spectrum.n_above_noise <= 1


def test_the_noise_edge_separates_real_factors_from_noise():
    rng = np.random.default_rng(3)
    factors = rng.normal(0, 0.01, (1500, 3))
    strategies = factors @ rng.normal(0, 1, (3, 60)) + rng.normal(0, 0.01, (1500, 60))
    assert rmt.spectrum(strategies).n_above_noise == 3


def test_reality_check_rejects_pure_noise_at_about_the_nominal_rate():
    p_values = []
    for seed in range(120):
        X = np.random.default_rng(seed).normal(0, 0.01, (500, 15))
        best = (X.mean(0) / X.std(0, ddof=1)).max()
        p_values.append(stats.reality_check(X, best, n_boot=150, seed=seed).p_value)
    p_values = np.array(p_values)
    assert (p_values < 0.05).mean() < 0.10  # rarely fooled by noise
    assert 0.3 < (p_values < 0.5).mean() < 0.7  # p-values roughly uniform under the null


def test_reality_check_finds_a_real_edge_and_ignores_duplicated_configurations():
    X = np.random.default_rng(1).normal(0, 0.01, (1500, 30))
    X[:, 4] += 0.0012
    best = (X.mean(0) / X.std(0, ddof=1)).max()
    once = stats.reality_check(X, best, seed=3)
    three_times = stats.reality_check(np.hstack([X, X, X]), best, seed=3)
    assert once.p_value < 0.01
    assert once.p_value == three_times.p_value  # pressing the button again is not more research


def test_exact_repeats_are_counted_once_even_after_storage():
    X = np.random.default_rng(4).normal(0, 0.01, (300, 6))
    stored = X.astype(np.float32).astype(float)  # how the court record keeps returns on disk
    distinct = stats.distinct_columns(np.hstack([stored, X, X[:, :2]]))
    assert distinct.shape == (300, 6)
    assert np.array_equal(distinct, stored)  # first occurrences, in their original order
