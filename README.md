# ⚖️ Overfit Court

[![tests](https://github.com/ekjots1ngh/overfit-court/actions/workflows/tests.yml/badge.svg)](https://github.com/ekjots1ngh/overfit-court/actions/workflows/tests.yml)

**AI agents can now try a thousand trading ideas before lunch. The more they try, the more they fool themselves. Overfit Court is the part that keeps count.**

Describe a strategy in plain English. NVIDIA Nemotron models on Nebius Token Factory plan it, write it and run every variant in parallel Token Factory Sandboxes. Then four things no ordinary backtester does:

1. **The time-travel test.** Every configuration is re-run on history cut short at up to fourteen points, from the first two days to the first 95%. If a position on a past date changes when later data arrives, the code read the future. The leak goes back to Nemotron Super, with the exact dates, for repair. The test fails closed: code that can't be re-run on shorter history, or answers differently when run twice on the same data, is treated as if it reads the future.
2. **The court remembers.** Every configuration ever tried on the data is kept, and a bootstrap Reality Check asks how often zero-edge versions of *all of them* would do as well as your pick.
3. **Random matrix theory counts your real bets.** 460 configurations might be 22 genuinely different bets. The eigenvalues of everything tried give the count, once the correlation that unrelated strategies show by chance has been removed, and that chance correlation is measured from the data, because shared volatility regimes inflate it.
4. **Sealed evidence.** The last 25% of the data is never uploaded to a research sandbox. Commit to one strategy and it gets exactly one test there, in a fresh sandbox, with the time-travel test run again at every sealed day. The commitment, a SHA-256 hash of the code, parameters, data, trading costs and scoring harness, is recorded before the seal is opened.

The statistics decide the verdict: **convicted, hung jury or acquitted.** The judge model only explains it.

Built for the Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering track). **Try it:** the app's **▶ Start here** tab tells the whole story in three clicks.

## Result 1: send an AI fisherman into pure noise

Sandboxes that branch like git make it almost free for an agent to try path after path until something works. Statisticians call that the garden of forking paths. A fisherman agent is told to find a strategy with Sharpe ≥ 1 and given 40 attempts. Each attempt is a new branch: a strategy family (8 of them), parameters, and a choice of asset universe ("the effect only works on these stocks"). It stops the moment it hits the target, like a real p-hacker. Then four judges look at what it brings back.

60 expeditions on pure noise and 60 on markets with a planted real edge:

| Market | Median branches | Median distinct bets | Hit Sharpe ≥ 1 | Standard test says real | Claim's-grid check says real | **Overfit Court acquits** | **Sealed evidence confirms** | **Both say real** |
|---|---|---|---|---|---|---|---|---|
| Pure noise | 460 | 22 | 8% | **82%** | 28% | **0%** | 3% | **0%** |
| Planted edge | 24 | 9 | 100% | 100% | 100% | **90%** | 43% | **37%** |

On data with nothing in it, a standard backtest significance test certified the fisherman's strategy as real **82% of the time**. Checking only the grid of the attempt it reported (selective reporting) still passed 28%. Overfit Court acquitted none. The sealed evidence is a single test at the 5% level, so on its own it confirmed 2 of the 60 noise claims, as such a test will; the court had not acquitted either of them, so nothing got past both. When there really is an edge, the fisherman finds it within a few attempts and the court acquits 90%.

Reproduce: `python -m examples.fishing_experiment 60` (about 15 minutes on a laptop). With `--nemotron` and a key, Nemotron 3 Super does the fishing instead of the scripted fisherman.

## Result 2: statistics cannot catch a strategy that reads the future

*"Buy assets trading below their long-run average price."* A plausible idea, and its natural first draft computes the average and spread over the whole series, including dates that haven't happened yet. A code review usually misses it. On 20 pure-noise markets:

| | Result |
|---|---|
| Median backtest Sharpe of the leaky version | 1.59 |
| Median Sharpe luck alone would give | 0.24 |
| **Court's statistics alone: acquitted** | **20 of 20** |
| **Time-travel test: caught** | **20 of 20** |
| Honest strategies wrongly flagged (every configuration of every library family, every market) | 0 of 180 (3,480 configurations, 48,720 checks) |

No amount of multiple-testing correction helps when the code has seen the answers. The time-travel test does: it re-runs every configuration on history truncated at up to fourteen points, from the first two days to 95% of them, and requires every earlier position to stay the same as data is added. It catches full-sample means, standard deviations and quantiles, negative shifts, back-fills (including a back-filled warm-up period, which only the early cuts can see) and centred windows, all tested. For each configuration the shorter histories run first, each with a freshly loaded strategy, and the full history runs last, twice. So its runs can't pass the future back to a shorter history, and a strategy that answers differently on the same data can't pass. In a trial, a leak is sent back to Nemotron Super with the parameters and dates involved, and the repaired code is re-tested before anything is judged. Code that still leaks, or can't be tested, is convicted and never allowed into the evidence room. There the test runs again with a cut at every sealed day, so even code that cheats for a few days, once it can see sealed dates, is caught (tested).

Reproduce: `python -m examples.time_travel_experiment 20`.

## Result 3: calibration on known truth

Single trials, judged on the open 75%, then the winner tested once on the sealed 25%. 20 random markets per level, cross-sectional momentum, 63 configurations, 5 bps costs.

| Planted edge | Convicted | Hung jury | Acquitted | Sealed: confirmed | Sealed: refuted | Median best Sharpe (open) | Median Sharpe (sealed) |
|---|---|---|---|---|---|---|---|
| none (noise) | 18 | 1 | 1 | 0 | 10 | 0.28 | 0.03 |
| weak | 7 | 6 | 7 | 4 | 5 | 0.79 | 0.51 |
| strong | 0 | 0 | 20 | 14 | 1 | 2.13 | 1.74 |

One noise market in 20 was acquitted on the open evidence, which is the 5% error a 95% test allows, and the sealed evidence refused to confirm it or any other noise strategy. That's the point of the evidence room: the open verdict is the discovery study, the sealed test is the replication. Reproduce: `python -m examples.calibration 20`.

## What we got wrong along the way

Every design choice was tested on synthetic markets where the right answer is known, and the code was reviewed cold, four times. Between them, they caught all of these.

- **The first version would have acquitted a cheat every time.** It had a regex and a model reviewing code for look-ahead bias. The full-sample-mean leak passes both, then sails through every statistical test (Result 2). That's why the time-travel test exists.
- **Then the time-travel test failed open.** A strategy that crashed on shorter history, or whose test branch was lost, counted as passing. And only three configurations per grid were tested, so a leak that switched on for one parameter value could win and walk free. Now every configuration is tested, and anything short of a complete pass counts as a failure (tested with exactly those strategies).
- **And it could still be dodged.** The second review found three ways past it. A strategy could keep its full-sample mean in a module-level cache, so every shorter run quietly reused the future; now every run loads the strategy afresh. The earliest cut was 1% of the data, too late to see a back-filled warm-up of a few days; now there are cuts at the first two, three, five and ten days. And the cuts stopped at 80%, so code that only started cheating late, once it could see sealed dates, was never caught; now the cuts reach 95%, and the evidence room added one every 2% across the sealed period. Each of those strategies is in the test suite.
- **And the evidence room still had gaps.** The third review showed that a cut every 2% of the history leaves gaps of several weeks. On one market, a strategy that read tomorrow's prices for about eight weeks inside one gap passed, and turned an honest REFUTED into CONFIRMED. The room now cuts at every sealed day. The same review found a strategy could still stash the future in another module or read it from the data file beside it, and that a trial failing halfway through left nothing on the record. Now shorter histories run before longer ones, no strategy code ever runs where the data file is, code that reaches outside its prices goes back for repair and is kept out of the evidence room, and a trial that fails is recorded as a mistrial. All tested.
- **And checking every sealed day was slow.** A final review found that an honest strategy written as a slow day-by-day loop now timed out in the evidence room, and the opening still counted. The room's hundreds of checks are now shared out across the sandbox branches, so they run in parallel, and every branch's returns must agree.
- **"Pure noise" drifted upwards.** Its log returns averaged zero, but the harness trades simple returns, which then averaged about 2% a year, so long-leaning strategies looked slightly skilful on data with nothing in it. Fixed, and every number in this README was re-run afterwards.
- **The distinct-bets count was biased low.** It removed sampling noise pair by pair, which undercounts (400 unrelated strategies counted as 287), and it assumed chance correlation is 1/T. On a market with volatility regimes, unrelated strategies look about twice as correlated as that, the same effect the author's earlier research found manufactures spurious factors. The noise level is now measured from the data, and 200 unrelated strategies count as 200 (tested).
- **The analytic Deflated Sharpe Ratio penalised pressing the button twice.** It counts configurations, so lookback 60 and lookback 61 count as separate discoveries, and re-running the identical grid looked like more fishing. The verdict uses a stationary block bootstrap Reality Check instead, which handles correlated configurations, fat tails and volatility clustering directly, and repeats of a configuration are counted once: running the same grid three times leaves every statistic unchanged (tested).
- **PBO convicted a genuine edge.** The Probability of Backtest Overfitting convicted a strategy whose in-sample winner made money out of sample in 99.6% of splits, because when every configuration is genuinely profitable, which one wins in-sample is random. The verdict now asks whether the in-sample winner *loses money* out of sample. PBO is still reported, as "do the exact parameters matter".
- **"P(real)" wasn't the probability the edge is real.** It was one minus a p-value, which is a different thing. The court now reports the p-value and says what it means.

## Honest limits

- **Lucky early hits can't be told from skill.** If a fisherman hits the target on its first attempt or two, the court has seen only a dozen configurations, and a 5% test lets a lucky dozen through about one time in twenty. It didn't happen in Result 1's 60 noise expeditions, but it does happen: one of Result 3's 20 noise markets was acquitted, and the sealed evidence refused to confirm it. That's what a 5% test allows, and it's why the sealed evidence exists.
- **The sealed test has modest power.** About two and a half years of sealed data confirms 43% of the fishermen's genuine edges (Result 1) and 70% of the strong ones in Result 3. It's built to almost never confirm a fake, not to always confirm a real one.
- **The time-travel test sees leaks that move a past position.** Knowledge written into the code itself, such as a hard-coded crash date, changes nothing when history is cut, so it gets past the test. On synthetic data there is nothing to know in advance. On real data a model may remember what markets did before its training cutoff, so only sealed dates after that cutoff are truly unseen by it.
- **It is built for honest mistakes, not for an adversary.** The research phase samples up to fourteen cuts, which catches accidental leaks because they touch most days. Code written to cheat, between two cuts or by passing state from one configuration to the next, could fool the open verdict. It can't fool the evidence room, which checks one configuration at every sealed day in fresh sandboxes. Python can't wall off code that is determined to reach into the interpreter, so the audit flags the usual ways in (imports beyond numpy and pandas, files, frames, Python internals, changes to shared modules) and the evidence room stays shut for them.
- **The court's memory is only as good as its record.** The app keeps a record for your browser session, and has a button to wipe it; the command line keeps a permanent one in `~/.overfit_court/`. The record is per dataset, so any change to the data starts a new one. Like the time-travel test, it is built to keep honest research honest; what makes a result checkable by someone else is the court record and its commitment hash.
- **Synthetic markets.** Calibration needs known truth, so the headline numbers use synthetic regime-switching markets. Real data works the same way (`examples/fetch_prices.py`, or upload a CSV in the app).

## How a trial works

1. **Planner (Nemotron 3 Ultra)** turns the idea into a hypothesis, a parameter grid and prior-art queries. The spec is validated (at least two configurations, numeric values, valid parameter names) and returned once with the exact problem if it's malformed.
2. **Clerk (Tavily + Nemotron 3 Nano)** searches for published versions of the effect and triages them. Only citations the search actually returned are kept, and a failed search never stops a trial.
3. **Engineer (Nemotron 3 Super)** writes `positions(prices, **params)`.
4. **Token Factory Sandboxes** run the grid on the open evidence only, one branch per chunk of the grid, and every configuration is time-travel tested inside its branch. The harness applies the one-day lag and costs itself, so generated code cannot trade on the close it just saw. No strategy code runs where the data file is: the checkpoint only checks the code is valid Python, and each branch deletes the file before loading the strategy. A branch lost to a network error is retried once; if it is lost again, the trial is a mistrial rather than the code's fault.
5. **Repair loop:** crashes, parameters that change nothing, time-travel failures and code that reaches outside its prices go back to Nemotron Super with the error or the dates, up to twice. If a repair breaks code that used to run, the last version that ran is judged. Code that still reads the future, or still can't be verified, is convicted; code that never runs at all is a mistrial. Either way the configurations tried go on the record.
6. **Auditor (static rules + Nemotron 3 Nano)** reviews the code judged. An answer it can't read counts as medium risk, not low. Code that still reaches outside the prices it is given (imports beyond numpy and pandas, files, Python internals, changes to shared modules) is high risk, and the evidence room stays shut for it.
7. **Statistics:** the Reality Check p-value over the whole court record, RMT distinct bets, CSCV (how often the in-sample winner loses out of sample), walk-forward selection, and "years of live trading to prove it": how long a live record at the walk-forward Sharpe would take to show the Sharpe is above zero.
8. **Judge (Nemotron 3 Ultra)** explains the ruling and is told to use only numbers in the evidence. A headline that contradicts the verdict is replaced with the court's own, and if the reply can't be read the court writes its own opinion. Either way the record says so.
9. **Evidence room:** commit (recorded first, with the count of earlier openings), then one test on sealed data in a fresh sandbox, time-travel tested at every sealed day.
10. **Court record:** a self-contained HTML page with the verdict, the numbers, the time-travel result and the commitment hash, to attach to a write-up or application.

| Verdict | Rule |
|---|---|
| CONVICTED | It reads the future or can't be shown not to; or luck matches it more often than not (p > 0.5); or walk-forward Sharpe ≤ 0; or the in-sample winner loses out of sample in ≥ 50% of splits |
| ACQUITTED | p ≤ 0.05, walk-forward Sharpe > 0, winner loses out of sample in < 20% of splits, look-ahead risk not high |
| HUNG JURY | anything in between |
| Sealed evidence | CONFIRMED if a one-sided test that the sealed Sharpe is above zero gives p ≤ 0.05, corrected for every earlier opening; REFUTED if the sealed Sharpe ≤ 0; else INCONCLUSIVE |

## Sandboxes as an epistemic guarantee

```
python:3.12-slim ──> + numpy, pandas                       (cached, same versions as the launching machine)
                        ├──> + OPEN data + strategy         (research checkpoint)
                        │       ├──> branch 1: configs 1, 9, 17, ...   (each backtested and time-travel tested)
                        │       ├──> branch 2: configs 2, 10, 18, ...  ... 8 branches in parallel microVMs
                        │       └──> ...
                        └──> + FULL data + ONE committed strategy   (evidence room, after commitment)
                                ├──> branch 1: sealed days 1, 9, 17, ...   (time-travel checks, in parallel)
                                └──> ...
```

Isolation usually means "the agent can't break my machine". Here it also means "the agent can't have peeked": the sealed dates are not in any research checkpoint, and a test proves every dataset handed to a research executor ends before the seal. One harness file (`overfit_court/_harness.py`) is uploaded verbatim as each sandbox's runner and imported locally, and the sandbox image gets exactly the numpy and pandas versions of the machine that launched it, so lags, costs and the time-travel test are identical everywhere. The executor is written against contree-sdk's real API (read from its source) and tested with a stand-in that executes the commands, producing returns identical to local execution.

## Model routing, built for live models

| Role | Tier | Why this tier |
|---|---|---|
| Planner | Nemotron 3 Ultra | Turning a vague idea into a falsifiable spec needs real reasoning |
| Engineer | Nemotron 3 Super | Strong code generation, and repair from a precise error, at a fraction of Ultra's price |
| Fisherman | Nemotron 3 Super | Decides where to fish next from the history of attempts |
| Auditor | Nemotron 3 Nano | Short focused review of the code judged |
| Clerk | Nemotron 3 Nano | Classifying search snippets is cheap work |
| Judge | Nemotron 3 Ultra | Must explain statistics faithfully without inventing numbers |

Every request, including retries and failures, is logged with tokens, latency and cost. The client expects real models to misbehave: if a request in JSON mode is rejected and works without it, the client asks for JSON in the prompt from then on; if a reasoning model spends its budget thinking it retries with more room; reasoning tags are stripped; and an unparseable reply is re-asked once. Every reply is checked before use, and if the Nemotron fisherman proposes something unusable, the scripted fisherman takes that turn and the attempt says so. A test runs a whole trial through a fake Token Factory that answers like a real model: reasoning tags, prose around the code, a sloppy parameter grid and a leaky first draft.

## Quickstart

```bash
git clone https://github.com/ekjots1ngh/overfit-court && cd overfit-court
pip install -r requirements.txt
cp .env.example .env                 # NEBIUS_API_KEY, NEBIUS_PROJECT_ID, TAVILY_API_KEY
python -m overfit_court.preflight    # checks models, JSON mode, sandboxes and Tavily end to end
streamlit run app.py
```

Without keys the app runs in demo mode: model replies are canned and ideas are routed to a built-in strategy library, but the harness, time-travel test, statistics and evidence room are all real. With a key, backtests run in Token Factory Sandboxes by default, because model-written code then runs; `OC_EXECUTOR=local` runs them in subprocesses for development.

Real data: `pip install yfinance && python -m examples.fetch_prices sectors` writes `sectors.csv` (11 US sector ETFs), which the app and the command line accept:

```bash
python -m overfit_court "buy last quarter's winners, short the losers" --csv sectors.csv --unseal
python -m overfit_court fish --csv sectors.csv --target 1.0 --attempts 40
```

The command line keeps its court record in `~/.overfit_court/`, so every run counts against the next one on the same data.

Deploy: the `Dockerfile` serves the app on port 8501 (for example on Nebius Serverless Endpoints); CPU is enough, because the models run on Token Factory and the backtests in Token Factory Sandboxes.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q            # 134 tests, all offline, about six minutes
ruff check . && ruff format --check .
```

Every test runs offline: two small fakes stand in for Token Factory (`tests/fake_token_factory.py`, which can misbehave on purpose: rejecting JSON mode, running out of tokens mid-thought, wrapping JSON in prose) and for Token Factory Sandboxes (`tests/fake_contree.py`, which really executes each branch, so checkpoint-and-branch behaviour is tested for real). The suite covers the statistics on known truth, the time-travel test against eight kinds of leak, strategies that crash on short history or answer differently on the same data, a leak that only some parameter values switch on, cheats that start only on sealed dates or last only a few weeks, a future stashed in another module, strategies that try to read the data file or reach outside their prices, lost, retried and timed-out sandbox branches, an evidence room shared across branches, mistrials, and every configuration of every honest strategy; sealed-evidence isolation and commitments; the court's memory, including a damaged record, a held lock and two runs sharing one record; the repair loop; fishing expeditions; model replies in the wrong shape; pre-flight; the court records; and the app's demo acts. GitHub Actions runs lint, format and tests on every push.

### A short code tour

Start with `overfit_court/pipeline.py`: `run_trial` reads as six numbered steps, and everything else hangs off it.

1. `_harness.py` is the single source of truth for scoring a strategy: the one-day lag, costs, and the time-travel test. It is uploaded verbatim into every sandbox.
2. `stats.py` holds every statistical test, in the order a trial uses them, with references. `verdict_from_stats` is the whole ruling, and its docstring says why each check is there.
3. `rmt.py` counts distinct bets; its docstring explains the noise correction.
4. `sandbox.py` runs the harness in Token Factory Sandboxes (one branch per chunk of the grid), in local subprocesses, or in-process for trusted library code.
5. `agents.py` holds the five model roles and their prompts; none of them decides the verdict, and every reply is checked before use.

## Nebius and NVIDIA usage

- **Nebius Token Factory**: every model call, through the OpenAI-compatible API (`overfit_court/llm.py`).
- **NVIDIA Nemotron 3 Nano, Super and Ultra**: routed by role as above.
- **Token Factory Sandboxes** (contree-sdk): checkpointed, branched execution of model-written code, the time-travel test inside every branch, and the evidence room (`overfit_court/sandbox.py`).
- **Tavily**: runtime prior-art search (`overfit_court/research.py`).

## Project layout

```
overfit_court/
  _harness.py   the one harness: lag, costs, time-travel test (uploaded into every sandbox)
  stats.py      Reality Check, DSR/PSR, CSCV, walk-forward, min track record, verdict
  rmt.py        correlation spectrum, noise edge, distinct bets
  ledger.py     the court record: every configuration's returns, every seal opening
  pipeline.py   trials, repair loop, sealed evidence split, evidence room, commitments
  fishing.py    the fishing expedition, scripted and Nemotron fishermen
  library.py    strategy families, the universe degree of freedom, the leaky demo draft
  agents.py     planner, engineer, auditor, clerk, judge
  llm.py        Token Factory client: tier routing, call log, live-model fallbacks
  sandbox.py    executors: Token Factory Sandboxes, local subprocess, in-process (trusted only)
  stories.py    the three demo acts
  report.py     court records: a shareable HTML page, and Markdown for the CLI
  research.py   Tavily prior-art search
  preflight.py  live setup check: models, JSON mode, a real sandbox job, Tavily
  config.py     settings from environment variables (see .env.example)
  data.py       synthetic markets with known truth, CSV loading
app.py          Streamlit UI: Start here, trial, fishing expedition, how it works
examples/       the experiments behind every number in this README, real data download
tests/          offline tests, with fakes for Token Factory and Sandboxes
docs/           feedback on the Nebius and NVIDIA tools
```

## Built during the submission period

All code was written after 26 August 2026 for this hackathon. The distinct-bets method is motivated by the author's earlier independent research on spurious factors in nonstationary markets (random matrix theory), but no code was reused.

## Feedback on Nebius and NVIDIA tools

See [docs/FEEDBACK.md](docs/FEEDBACK.md).

## Licence

MIT, see [LICENSE](LICENSE).
