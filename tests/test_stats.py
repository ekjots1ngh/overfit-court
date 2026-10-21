import numpy as np
import pytest

from overfit_court import stats
from overfit_court.stats import ACQUITTED, CONVICTED, HUNG_JURY


def test_psr_is_one_half_when_the_sharpe_equals_the_benchmark():
    assert stats.probabilistic_sharpe_ratio(0.05, 0.05, 1000) == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    assert 0 < stats.expected_max_sharpe(0.01, 10) < stats.expected_max_sharpe(0.01, 1000)


def test_the_best_of_many_noise_strategies_is_deflated():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, size=(2520, 200))
    sharpes = [stats.sharpe(M[:, j]) for j in range(M.shape[1])]
    best = int(np.argmax(sharpes))
    assert stats.probabilistic_sharpe_ratio(sharpes[best], 0.0, M.shape[0]) > 0.9  # looks great
    assert stats.deflated_sharpe_ratio(M[:, best], sharpes) < 0.5  # until corrected for 200 tries


def test_pbo_is_near_one_half_for_pure_noise():
    M = np.random.default_rng(1).normal(0, 0.01, size=(2000, 50))
    assert 0.25 < stats.pbo_cscv(M, n_splits=10).pbo < 0.75


def test_pbo_is_low_when_one_configuration_has_a_real_edge():
    M = np.random.default_rng(2).normal(0, 0.01, size=(2000, 50))
    M[:, 7] += 0.0015
    assert stats.pbo_cscv(M, n_splits=10).pbo < 0.1


def test_walk_forward_picks_the_real_edge():
    M = np.random.default_rng(3).normal(0, 0.01, size=(2520, 30))
    M[:, 4] += 0.0015
    result = stats.walk_forward(M, n_folds=6)
    assert result.oos_sharpe_annual > 1.0
    assert result.selected_columns.count(4) >= 4


def test_min_track_record_needs_longer_for_a_thinner_edge():
    assert stats.min_track_record_years(0.02, 0.05) is None  # doesn't beat the benchmark
    assert 0 < stats.min_track_record_years(0.2, 0.05) < stats.min_track_record_years(0.07, 0.05)


@pytest.mark.parametrize(
    ("p_real", "walk_forward", "oos_loss", "expected"),
    [
        (0.30, 1.0, 0.10, CONVICTED),  # luck explains it
        (0.99, -0.2, 0.10, CONVICTED),  # loses money chosen walk-forward
        (0.99, 1.0, 0.60, CONVICTED),  # the in-sample winner loses out of sample
        (0.99, 1.0, 0.05, ACQUITTED),
        (0.90, 1.0, 0.05, HUNG_JURY),  # promising, not proven
    ],
)
def test_verdict_rules(p_real, walk_forward, oos_loss, expected):
    assert stats.verdict_from_stats(p_real, walk_forward, oos_loss) == expected


def test_probabilities_are_never_rounded_to_certainty():
    assert stats.fmt_prob(0.997) == "over 99%"
    assert stats.fmt_prob(0.001) == "under 1%"
    assert stats.fmt_prob(0.42) == "42%"
