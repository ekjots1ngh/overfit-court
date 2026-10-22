"""The court remembers, and never silently forgets."""

import numpy as np
import pytest
from filelock import FileLock

from overfit_court import ledger as ledger_module
from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.ledger import Ledger, LedgerError, fingerprint, record_key
from overfit_court.pipeline import run_trial
from overfit_court.sandbox import InProcessExecutor


def trial_on(prices, ledger, settings, silent, idea="momentum"):
    return run_trial(
        idea, prices, settings=settings, ledger=ledger, executor=InProcessExecutor(), log=silent
    )


def test_fingerprints_identify_the_exact_data():
    assert fingerprint(synthetic_prices(seed=1)) != fingerprint(synthetic_prices(seed=2))
    assert fingerprint(synthetic_prices(seed=1)) == fingerprint(synthetic_prices(seed=1))


def test_rerunning_the_same_grid_is_not_penalised(tmp_path, settings, silent):
    """Pressing the button twice, or three times, is not more research."""
    prices = synthetic_prices(seed=7, signal=PLANTED_SIGNAL)
    ledger = Ledger(tmp_path)
    first = trial_on(prices, ledger, settings, silent).evidence()
    trial_on(prices, ledger, settings, silent)
    third = trial_on(prices, ledger, settings, silent).evidence()
    assert third["trial_number_on_this_data"] == 3
    assert third["n_configs_tried_on_this_data"] == 3 * first["n_configs"]
    assert third["n_distinct_configs_on_this_data"] == first["n_configs"]
    for number in ("p_value", "distinct_bets_total", "prob_oos_loss", "walk_forward_sharpe"):
        assert third[number] == first[number], number


def test_new_ideas_raise_the_bar_and_the_record_survives_a_restart(tmp_path, settings, silent):
    prices = synthetic_prices(seed=7)
    ledger = Ledger(tmp_path)
    first = trial_on(prices, ledger, settings, silent)
    for idea in ["buy on mondays", "golden cross", "calm stocks", "new highs"]:
        last = trial_on(prices, ledger, settings, silent, idea)
    e = last.evidence()
    assert e["distinct_bets_total"] > first.evidence()["distinct_bets_total"] + 2
    assert e["luck_sharpe_annual"] > e["luck_sharpe_annual_this_trial_only"]

    reopened = Ledger(tmp_path)
    key = record_key(prices, 0.25)
    assert [h["family"] for h in reopened.history(key)] == [
        "momentum",
        "weekday",
        "ma_cross",
        "low_vol",
        "breakout",
    ]
    tried = sum(array.shape[1] for array in reopened.prior_returns(key))
    assert tried == e["n_configs_tried_on_this_data"]


def test_what_the_ledger_hands_out_never_changes_behind_your_back():
    ledger, key = Ledger(), "k"
    ledger.record(key, "first", np.ones((10, 2)), "CONVICTED")
    before, history = ledger.prior_returns(key), ledger.history(key)
    ledger.record(key, "second", np.ones((10, 3)), "CONVICTED")
    assert len(before) == 1 and len(history) == 1
    assert len(ledger.prior_returns(key)) == 2


def test_a_damaged_record_is_an_error_not_a_fresh_start(tmp_path, settings, silent):
    prices = synthetic_prices(seed=7)
    trial_on(prices, Ledger(tmp_path), settings, silent)
    key = record_key(prices, 0.25)

    next(tmp_path.glob("*.npy")).unlink()
    with pytest.raises(LedgerError, match="missing"):
        Ledger(tmp_path).prior_returns(key)

    (tmp_path / "index.json").write_text('{"truncated": ')
    with pytest.raises(LedgerError, match="can't be read"):
        Ledger(tmp_path)
    assert (tmp_path / "index.json").read_text() == '{"truncated": '  # left as it was


def test_two_runs_on_one_record_never_overwrite_each_other(tmp_path):
    first, second = Ledger(tmp_path), Ledger(tmp_path)  # both opened before either writes
    first.record("k", "first idea", np.ones((10, 2)), "CONVICTED")
    second.record("k", "second idea", np.ones((10, 3)), "CONVICTED")
    reopened = Ledger(tmp_path)
    assert [trial["idea"] for trial in reopened.history("k")] == ["first idea", "second idea"]
    assert [array.shape[1] for array in reopened.prior_returns("k")] == [2, 3]


def test_every_read_sees_what_other_runs_have_recorded(tmp_path):
    """A trial running alongside a fishing expedition on the same data counts its branches."""
    reader, writer = Ledger(tmp_path), Ledger(tmp_path)
    assert reader.history("k") == [] and reader.prior_returns("k") == []
    writer.record("k", "the expedition", np.ones((10, 40)), "CONVICTED")
    assert writer.record_seal_opening("k", "first", {}) == 0
    assert [trial["idea"] for trial in reader.history("k")] == ["the expedition"]
    assert [array.shape[1] for array in reader.prior_returns("k")] == [40]
    assert reader.record_seal_opening("k", "second", {}) == 1  # counted under the lock


def test_a_held_lock_is_reported_not_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "LOCK_TIMEOUT_S", 0.2)
    with FileLock(tmp_path / "index.lock"), pytest.raises(LedgerError, match="locked"):
        Ledger(tmp_path).record("k", "idea", np.ones((10, 2)), "CONVICTED")


def test_a_lock_file_left_by_a_killed_run_is_harmless(tmp_path):
    (tmp_path / "index.lock").write_text("")  # the run died; its operating-system lock went
    Ledger(tmp_path).record("k", "idea", np.ones((10, 2)), "CONVICTED")
    assert len(Ledger(tmp_path).history("k")) == 1


def test_a_cleared_record_never_serves_old_returns(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record("k", "old", np.ones((10, 2)), "CONVICTED")
    ledger.prior_returns("k")  # read once, so it is cached
    ledger.clear("k")
    ledger.record("k", "new", np.zeros((10, 3)), "CONVICTED")
    [returns] = ledger.prior_returns("k")
    assert returns.shape == (10, 3) and not returns.any()


def test_a_record_reopened_from_disk_gives_exactly_the_same_numbers(tmp_path):
    returns = np.random.default_rng(0).normal(0, 0.01, (50, 3))
    in_memory, on_disk = Ledger(), Ledger(tmp_path)
    for ledger in (in_memory, on_disk):
        ledger.record("k", "idea", returns, "CONVICTED")
    [kept] = in_memory.prior_returns("k")
    [reloaded] = Ledger(tmp_path).prior_returns("k")
    assert np.array_equal(kept, reloaded)


def test_clearing_forgets_the_data_and_its_files(tmp_path, settings, silent):
    prices = synthetic_prices(seed=7)
    ledger = Ledger(tmp_path)
    trial_on(prices, ledger, settings, silent)
    key = record_key(prices, 0.25)
    ledger.clear(key)
    assert ledger.history(key) == []
    assert list(tmp_path.glob("*.npy")) == []
