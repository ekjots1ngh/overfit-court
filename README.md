# ⚖️ Overfit Court

**AI agents can now try a thousand trading ideas before lunch. The more they try, the more they fool themselves. Overfit Court is the part that keeps count.**

Describe a strategy in plain English. NVIDIA Nemotron models on Nebius Token Factory plan it, write it and run every variant in parallel Token Factory Sandboxes. Then four things no ordinary backtester does:

1. **The time-travel test.** Every trial re-runs the strategy on history cut short at several dates. If a position on a past date changes when later data arrives, the code read the future. The leak goes back to Nemotron Super, with the exact dates, for repair.
2. **The court remembers.** Every configuration ever tried on the data is kept, and a bootstrap Reality Check asks how good the best of *all of them* would look if none had any edge.
3. **Random matrix theory counts your real bets.** 460 configurations might be 22 genuinely different bets. The eigenvalue spectrum of everything tried is compared with the Marchenko-Pastur law to find out.
4. **Sealed evidence.** The last 25% of the data is never uploaded to a research sandbox. Commit to one strategy (a SHA-256 commitment is recorded first) and it gets exactly one test there, in a fresh sandbox.

The statistics decide the verdict: **convicted, hung jury or acquitted.** The judge model only explains it.

Built for the Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering track). **Try it:** the app's **▶ Start here** tab tells the whole story in three clicks.

## Result 1: send an AI fisherman into pure noise

Sandboxes that branch like git make it almost free for an agent to try path after path until something works. Statisticians call that the garden of forking paths. A fisherman agent is told to find a strategy with Sharpe ≥ 1 and given 40 attempts. Each attempt is a new branch: a strategy family (8 of them), parameters, and a choice of asset universe ("the effect only works on these stocks"). It stops the moment it hits the target, like a real p-hacker. Then four judges look at what it brings back.

60 expeditions on pure noise and 60 on markets with a planted real edge:

| Market | Median branches | Median distinct bets | Hit Sharpe ≥ 1 | Standard test says real | Final-grid check says real | **Overfit Court acquits** | **Sealed evidence confirms** |
|---|---|---|---|---|---|---|---|
| Pure noise | 461 | 22 | 12% | **83%** | 27% | **0%** | **0%** |
| Planted edge | 24 | 8 | 98% | 100% | 98% | **85%** | **47%** |

On data with nothing in it, a standard backtest significance test certified the fisherman's strategy as real **83% of the time**. Checking only the final grid it reported (selective reporting) still passed 27%. Overfit Court acquitted none, and the sealed evidence confirmed none. When there really is an edge, the fisherman finds it within a few attempts and the court acquits 85%.

Reproduce: `python examples/fishing_experiment.py 60` (about 12 minutes). With `--nemotron` and a key, Nemotron 3 Super does the fishing instead of the scripted fisherman.

## Result 2: statistics cannot catch a strategy that reads the future

*"Buy assets trading below their long-run average price."* A plausible idea, and its natural first draft computes the average and spread over the whole series, including dates that haven't happened yet. A code review usually misses it. On 20 pure-noise markets:

| | Result |
|---|---|
| Median backtest Sharpe of the leaky version | 1.63 |
| Median Sharpe luck alone would give | 0.24 |
| **Court's statistics alone: acquitted** | **20 of 20** |
| **Time-travel test: caught** | **20 of 20** |
| Honest strategies wrongly flagged (every library family, every market) | 0 of 180 |

No amount of multiple-testing correction helps when the code has seen the answers. The time-travel test does: it runs the strategy on history truncated at 40%, 60% and 80% and requires every earlier position to be identical to the full run. It catches full-sample means, standard deviations and quantiles, negative shifts, back-fills and centred windows (all tested). In a trial, a leak is sent back to Nemotron Super with the parameters and dates involved; the repaired code is re-tested before anything is judged, and code that still leaks is convicted and never allowed into the evidence room.

Reproduce: `python examples/time_travel_experiment.py 20`.

## Result 3: calibration on known truth

Single trials, judged on the open 75%, then the winner tested once on the sealed 25%. 20 random markets per level, cross-sectional momentum, 63 configurations, 5 bps costs.

| Planted edge | Convicted | Hung jury | Acquitted | Sealed: confirmed | Sealed: refuted | Median best Sharpe (open) | Median Sharpe (sealed) |
|---|---|---|---|---|---|---|---|
| none (noise) | 17 | 2 | 1 | 0 | 10 | 0.33 | 0.07 |
| weak | 4 | 8 | 8 | 5 | 4 | 0.83 | 0.55 |
| strong | 0 | 0 | 20 | 14 | 1 | 2.17 | 1.78 |

One noise market in 20 was acquitted on open evidence, which is the 5% error a 95% test allows, and the sealed evidence refused to confirm it or any other noise strategy. That's the point of the evidence room: the open verdict is the discovery study, the sealed test is the replication. Reproduce: `python examples/calibration.py 20`.

## What we got wrong along the way

Every design choice was tested on synthetic markets where the right answer is known. Three of ours failed.

- **The first version would have acquitted a cheat every time.** It had a regex and a model reviewing code for look-ahead bias. The full-sample-mean leak passes both, then sails through every statistical test (Result 2). That's why the time-travel test exists.
- **The analytic Deflated Sharpe Ratio penalised pressing the button twice.** It counts configurations, so lookback 60 and lookback 61 count as separate discoveries, and re-running the identical grid looked like more fishing. The stationary block bootstrap Reality Check handles correlated configurations, fat tails and volatility clustering directly, and duplicating a grid leaves it exactly unchanged (tested).
- **PBO convicted a genuine edge.** The Probability of Backtest Overfitting convicted a strategy whose in-sample winner made money out of sample in 99.6% of splits, because when every configuration is genuinely profitable, which one wins in-sample is random. The verdict now asks whether the in-sample winner *loses money* out of sample. PBO is still reported, as "do the exact parameters matter".

## Honest limits

- **Lucky early hits can't be told from skill.** In a separate search over 60 noise markets with differently seeded fishermen, the court acquitted 2. Both times the fisherman hit the target within its first five attempts, before fishing had raised the bar. That's what a 95% test allows; it's why the sealed evidence exists.
- **The sealed test has modest power.** About two and a half years of sealed data confirms roughly half of genuine edges (47% above). It's built to almost never confirm a fake, not to always confirm a real one.
- **The time-travel test sees leaks that cross a cut date.** A leak confined to the first rows of the data (for example back-filling the very first missing value) is invisible to it, and also can't affect trading, because the harness lags every position by a day.
- **Synthetic markets.** Calibration needs known truth, so the headline numbers use synthetic regime-switching markets. Real data works the same way (`examples/fetch_prices.py`, or upload a CSV in the app).

## How a trial works

1. **Planner (Nemotron 3 Ultra)** turns the idea into a hypothesis, a parameter grid and prior-art queries. The spec is validated and returned once with the exact problem if malformed.
2. **Clerk (Tavily + Nemotron 3 Nano)** searches for published versions of the effect and triages them.
3. **Engineer (Nemotron 3 Super)** writes `positions(prices, **params)`.
4. **Token Factory Sandboxes** run the grid on the open evidence only, plus the time-travel test as its own branch. The harness applies the one-day lag and costs itself, so generated code cannot trade on the close it just saw.
5. **Repair loop:** crashes and time travel go back to Nemotron Super with the error or the dates, up to twice. Code that still reads the future is convicted.
6. **Auditor (static rules + Nemotron 3 Nano)** reviews the final code.
7. **Statistics:** Reality Check over the whole court record, RMT distinct bets, CSCV (probability the in-sample winner loses out of sample), walk-forward selection, minimum track record length ("years of live trading to prove it").
8. **Judge (Nemotron 3 Ultra)** explains the ruling using only numbers in the evidence, and names the checks that failed.
9. **Evidence room:** commit, then one test on sealed data in a fresh sandbox.
10. **Court record:** a self-contained HTML page with the verdict, the numbers, the time-travel result and the commitment hash, to attach to a write-up or application.

| Verdict | Rule |
|---|---|
| CONVICTED | It reads the future, or P(real) < 50%, or walk-forward Sharpe ≤ 0, or the in-sample winner loses out of sample in ≥ 50% of splits |
| ACQUITTED | P(real) ≥ 95%, walk-forward Sharpe > 0, winner loses out of sample in < 20% of splits, look-ahead risk not high |
| HUNG JURY | anything in between |
| Sealed evidence | CONFIRMED if P(real) ≥ 95% on the sealed data (corrected for repeat openings); REFUTED if its Sharpe ≤ 0; else INCONCLUSIVE |

## Sandboxes as an epistemic guarantee

```
python:3.12-slim ──> + numpy, pandas                       (cached)
                        ├──> + OPEN data + strategy         (research checkpoint)
                        │       ├──> branch: configs 1-15
                        │       ├──> branch: configs 16-30   ... in parallel microVMs
                        │       └──> branch: time-travel test
                        └──> + FULL data + ONE committed strategy   (evidence room, after commitment)
```

Isolation usually means "the agent can't break my machine". Here it also means "the agent can't have peeked": the sealed dates are not in any research checkpoint, and a test proves every dataset handed to a research executor ends before the seal. One harness file (`overfit_court/_harness.py`) is uploaded verbatim as each sandbox's runner and imported locally, so lags, costs and the time-travel test are identical everywhere. The executor is written against contree-sdk's real API (read from its source) and tested with a stand-in that executes the commands, producing returns identical to local execution.

## Model routing, built for live models

| Role | Tier | Why this tier |
|---|---|---|
| Planner | Nemotron 3 Ultra | Turning a vague idea into a falsifiable spec needs real reasoning |
| Engineer | Nemotron 3 Super | Strong code generation, and repair from a precise error, at a fraction of Ultra's price |
| Fisherman | Nemotron 3 Super | Decides where to fish next from the history of attempts |
| Auditor | Nemotron 3 Nano | Short focused review of the final code |
| Clerk | Nemotron 3 Nano | Classifying search snippets is cheap work |
| Judge | Nemotron 3 Ultra | Must explain statistics faithfully without inventing numbers |

Every request, including retries, is logged with tokens, latency and cost. The client expects real models to misbehave: if a tier rejects JSON mode it falls back to asking in the prompt (and remembers), if a reasoning model spends its budget thinking it retries with more room, reasoning tags are stripped, and an unparseable reply is re-asked once. A test runs a whole trial through a fake Token Factory that answers like a real model: reasoning tags, prose around the code, a sloppy parameter grid and a leaky first draft.

<!-- LIVE NUMBERS: replace this line with the output of a live trial and expedition: cost per trial, cost per expedition, sandbox wall time vs local -->

## Quickstart

```bash
git clone https://github.com/ekjots1ngh/overfit-court && cd overfit-court
pip install -r requirements.txt
cp .env.example .env                 # NEBIUS_API_KEY, NEBIUS_PROJECT_ID, TAVILY_API_KEY
python -m overfit_court.preflight    # checks models, JSON mode, sandboxes and Tavily end to end
streamlit run app.py
```

Without keys the app runs in demo mode: model replies are canned and ideas are routed to a built-in strategy library, but the harness, time-travel test, statistics and evidence room are all real.

CLI:

```bash
python -m overfit_court "buy last quarter's winners, short the losers" --csv prices.csv --unseal
python -m overfit_court fish --csv prices.csv --target 1.0 --attempts 40
```

Real data: `pip install yfinance && python examples/fetch_prices.py sectors`.

Tests: `pytest -q` (41 tests: the time-travel test against five kinds of leak and every honest strategy, sealed-evidence isolation, Reality Check calibration and duplicate invariance, distinct-bets counting, the court's memory, fishing expeditions, the sandbox executor against a fake contree client, the live client against a misbehaving fake Token Factory, and a full trial through a live-like model).

Deploy: the `Dockerfile` serves the app on port 8501 (for example on Nebius Serverless Endpoints); CPU is enough.

## Nebius and NVIDIA usage

- **Nebius Token Factory**: every model call, through the OpenAI-compatible API (`overfit_court/llm.py`).
- **NVIDIA Nemotron 3 Nano, Super and Ultra**: routed by role as above.
- **Token Factory Sandboxes** (contree-sdk): checkpointed, branched execution of model-written code, the time-travel test as its own branch, and the evidence room (`overfit_court/sandbox.py`).
- **Tavily**: runtime prior-art search (`overfit_court/research.py`).

## Project layout

```
overfit_court/
  _harness.py   the one harness: lag, costs, time-travel test (uploaded into every sandbox)
  stats.py      Reality Check, DSR/PSR, CSCV, walk-forward, min track record, verdict
  rmt.py        correlation spectrum, Marchenko-Pastur band, distinct bets
  ledger.py     the court record: every configuration's returns, every seal opening
  pipeline.py   trials, repair loop, sealed evidence split, evidence room, commitments
  fishing.py    the fishing expedition, scripted and Nemotron fishermen
  library.py    strategy families, the universe degree of freedom, the leaky demo draft
  agents.py     planner, engineer, auditor, clerk, judge
  llm.py        Token Factory client: tier routing, cost ledger, live-model fallbacks
  sandbox.py    executors: Token Factory Sandboxes, local subprocess, in-process (trusted only)
  stories.py    the three demo acts
  report.py     the shareable HTML court record
  research.py   Tavily prior-art search
  preflight.py  live setup check
app.py          Streamlit UI: Start here, trial, fishing expedition, how it works
examples/       fishing, time-travel and calibration experiments, real data download
docs/           submission kit, feedback log
```

## Built during the submission period

All code was written after 26 August 2026 for this hackathon. The distinct-bets method is motivated by the author's earlier independent research on spurious factors in nonstationary markets (random matrix theory), but no code was reused.

## Feedback on Nebius and NVIDIA tools

See [docs/FEEDBACK.md](docs/FEEDBACK.md).

## Licence

MIT, see [LICENSE](LICENSE).
