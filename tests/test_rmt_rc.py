import numpy as np

from overfit_court import rmt, stats


def test_distinct_bets_counts_independent_groups():
    rng = np.random.default_rng(0)
    T = 1500
    indep = rng.normal(0, 0.01, (T, 40))
    assert 36 < rmt.spectrum(indep).distinct_bets < 41
    base = rng.normal(0, 0.01, (T, 5))
    groups = np.repeat(base, 8, axis=1) + rng.normal(0, 0.002, (T, 40))
    assert 4 < rmt.spectrum(groups).distinct_bets < 7
    assert abs(rmt.spectrum(np.hstack([indep, indep])).distinct_bets
               - rmt.spectrum(indep).distinct_bets) < 1.0


def test_reality_check_null_is_calibrated():
    ps = []
    for s in range(120):
        X = np.random.default_rng(s).normal(0, 0.01, (500, 15))
        srs = X.mean(0) / X.std(0, ddof=1)
        ps.append(stats.reality_check(X, srs.max(), n_boot=150, seed=s).p_value)
    ps = np.array(ps)
    assert (ps < 0.05).mean() < 0.10          # rejects noise rarely
    assert 0.3 < (ps < 0.5).mean() < 0.7      # roughly uniform


def test_reality_check_finds_real_edge_and_ignores_duplicates():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (1500, 30))
    X[:, 4] += 0.0012
    pick = (X.mean(0) / X.std(0, ddof=1)).max()
    a = stats.reality_check(X, pick, seed=3)
    b = stats.reality_check(np.hstack([X, X, X]), pick, seed=3)
    assert a.p_value < 0.01
    assert a.p_value == b.p_value
