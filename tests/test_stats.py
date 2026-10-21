import numpy as np
import pytest

from overfit_court import stats


def test_psr_is_half_when_sr_equals_benchmark():
    assert stats.probabilistic_sharpe_ratio(0.05, 0.05, 1000) == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_trials():
    a = stats.expected_max_sharpe(0.01, 10)
    b = stats.expected_max_sharpe(0.01, 1000)
    assert 0 < a < b


def test_best_of_many_noise_strategies_is_deflated():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, size=(2520, 200))
    srs = [stats.sharpe(M[:, j]) for j in range(M.shape[1])]
    best = int(np.argmax(srs))
    naive_psr = stats.probabilistic_sharpe_ratio(srs[best], 0.0, M.shape[0])
    dsr = stats.deflated_sharpe_ratio(M[:, best], srs).dsr
    assert naive_psr > 0.9          # looks great on its own
    assert dsr < 0.5                # but not after multiple-testing correction


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(1)
    M = rng.normal(0, 0.01, size=(2000, 50))
    res = stats.pbo_cscv(M, n_splits=10)
    assert 0.25 < res.pbo < 0.75


def test_pbo_low_when_one_config_has_real_edge():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, size=(2000, 50))
    M[:, 7] += 0.0015
    res = stats.pbo_cscv(M, n_splits=10)
    assert res.pbo < 0.1


def test_walk_forward_picks_the_real_edge():
    rng = np.random.default_rng(3)
    M = rng.normal(0, 0.01, size=(2520, 30))
    M[:, 4] += 0.0015
    wf = stats.walk_forward(M, n_folds=6)
    assert wf.oos_sharpe_annual > 1.0
    assert wf.selected_columns.count(4) >= 4


def test_verdicts():
    assert stats.verdict_from_stats(0.3, 1.0, 0.1) == "CONVICTED"      # luck explains it
    assert stats.verdict_from_stats(0.99, -0.2, 0.1) == "CONVICTED"    # fails walk-forward
    assert stats.verdict_from_stats(0.99, 1.0, 0.6) == "CONVICTED"     # winner loses OOS
    assert stats.verdict_from_stats(0.99, 1.0, 0.05) == "ACQUITTED"
    assert stats.verdict_from_stats(0.9, 1.0, 0.05) == "HUNG JURY"
