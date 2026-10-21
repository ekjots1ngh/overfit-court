"""Random matrix theory distinct bets, and the Reality Check."""

import numpy as np

from overfit_court import rmt, stats


def test_distinct_bets_counts_independent_groups():
    rng = np.random.default_rng(0)
    T = 1500
    independent = rng.normal(0, 0.01, (T, 40))
    assert 36 < rmt.spectrum(independent).distinct_bets < 41

    five_bets = rng.normal(0, 0.01, (T, 5))
    eight_copies_each = np.repeat(five_bets, 8, axis=1) + rng.normal(0, 0.002, (T, 40))
    assert 4 < rmt.spectrum(eight_copies_each).distinct_bets < 7

    duplicated = np.hstack([independent, independent])
    assert abs(rmt.spectrum(duplicated).distinct_bets - rmt.spectrum(independent).distinct_bets) < 1


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
