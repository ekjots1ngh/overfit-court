# ⚖️ Overfit Court

**AI agents can now try a thousand trading ideas before lunch. The more they try, the more they fool themselves. Overfit Court is the part that keeps count.**

Describe a strategy in plain English. NVIDIA Nemotron models on Nebius Token Factory plan it, write it and run every variant in parallel Token Factory Sandboxes. Then three things no ordinary backtester does:

1. **The court remembers.** Every configuration ever tried on the data is kept, and a bootstrap Reality Check asks how good the best of *all of them* would look if none had any edge.
2. **Random matrix theory counts your real bets.** 460 configurations might be 22 genuinely different bets. The eigenvalue spectrum of everything tried is compared with the Marchenko-Pastur law to find out.
3. **Sealed evidence.** The last 25% of the data is never uploaded to a research sandbox, so neither the model nor its code can have seen it. Commit to one strategy (a SHA-256 commitment is recorded first) and it gets exactly one test there, in a fresh sandbox.

The statistics decide the verdict: **convicted, hung jury or acquitted.** The judge model only explains it.

Built for the Nebius x NVIDIA Global AI Hackathon (Coding and Agentic Engineering track).

## The headline experiment: send an AI fisherman into pure noise

Sandboxes that branch like git make it almost free for an agent to try path after path until something works. Statisticians call that the garden of forking paths. We built a fisherman agent, told it to find a strategy with Sharpe ≥ 1, and gave it 40 attempts. Each attempt is a new branch: a strategy family (8 of them), parameters, and a choice of asset universe ("the effect only works on these stocks"). It stops the moment it hits the target, like a real p-hacker. Then four judges look at what it brings back.

20 expeditions on pure noise and 20 on markets with a planted real edge:

| Market | Median branches | Median distinct bets | Hit Sharpe ≥ 1 | Standard test says real | Final-grid check says real | **Overfit Court acquits** | **Sealed evidence confirms** |
|---|---|---|---|---|---|---|---|
| Pure noise | 460 | 22 | 10% | **80%** | 25% | **0%** | **0%** |
| Planted edge | 34 | 7 | 100% | 100% | 100% | **85%** | **50%** |

On data with nothing in it, a standard backtest significance test certified the fisherman's strategy as real **80% of the time**. Checking only the final grid it reported (selective reporting) still passed 25%. Overfit Court acquitted none, and the sealed evidence confirmed none. When there really is an edge, the fisherman finds it within a few attempts and the court acquits 85%.

Reproduce: `python examples/fishing_experiment.py 20`. With `--nemotron` and a key, Nemotron 3 Super does the fishing instead of the scripted fisherman.

## Calibration: single trials on known truth

Each trial is judged on the open 75%, then its winner is tested once on the sealed 25%. 20 random markets per level, cross-sectional momentum, 63 configurations, 5 bps costs.

| Planted edge | Convicted | Hung jury | Acquitted | Sealed: confirmed | Sealed: refuted | Median best Sharpe (open) | Median Sharpe (sealed) |
|---|---|---|---|---|---|---|---|
| none (noise) | 17 | 2 | 1 | 0 | 10 | 0.33 | 0.07 |
| weak | 4 | 8 | 8 | 5 | 4 | 0.83 | 0.55 |
| strong | 0 | 0 | 20 | 14 | 1 | 2.17 | 1.78 |

One noise market in 20 was acquitted on open evidence, which is the 5% error a 95% test allows, and the sealed evidence refused to confirm it or any other noise strategy. That is the point of the evidence room: the open verdict is the discovery study, the sealed test is the replication. Reproduce: `python examples/calibration.py 20`.

## Why these choices (and the ones we got wrong)

- **Reality Check, not a formula.** An earlier version used the analytic Deflated Sharpe Ratio with the raw number of configurations. That treats lookback 60 and lookback 61 as separate discoveries, and it penalised re-running the identical grid as if it were new research. The stationary block bootstrap handles correlated configurations, fat tails and volatility clustering directly, and duplicating a grid leaves it exactly unchanged (tested).
- **Distinct bets from random matrix theory.** Sample correlations are inflated by noise; the Marchenko-Pastur law says by how much. We subtract that before counting effective independent bets. 63 momentum variants come out as about 3 bets; adding four more strategy families added about 3 more, because several of them are the same market bet.
- **We dropped PBO from the verdict.** The Probability of Backtest Overfitting convicted a strategy whose in-sample winner made money out of sample in 99.6% of splits, because when every configuration is genuinely profitable, which one wins in-sample is random. The verdict now asks whether the in-sample winner loses money out of sample. PBO is still reported, as "do the exact parameters matter".
- **The sealed test corrects for repeat openings.** Every opening is logged; the p-value is Sidak-corrected by the number of times the seal has been broken.

## How a trial works

1. **Planner (Nemotron 3 Ultra)** turns the idea into a hypothesis, a parameter grid and prior-art queries. The spec is validated and returned once with the exact problem if malformed.
2. **Clerk (Tavily + Nemotron 3 Nano)** searches for published versions of the effect and triages them.
3. **Engineer (Nemotron 3 Super)** writes `positions(prices, **params)`; sandbox errors go back to Super for repair.
4. **Token Factory Sandboxes** run the grid on the open evidence only. The harness applies the one-day lag and costs itself, so generated code cannot trade on today's close.
5. **Auditor (static rules + Nemotron 3 Nano)** checks for look-ahead bias.
6. **Statistics:** Reality Check over the whole court record, RMT distinct bets, CSCV (probability the in-sample winner loses out of sample), walk-forward selection, and minimum track record length ("years of live trading to prove it").
7. **Judge (Nemotron 3 Ultra)** explains the ruling using only numbers in the evidence.
8. **Evidence room:** commit, then one test on sealed data in a fresh sandbox.

| Verdict | Rule |
|---|---|
| CONVICTED | P(real) < 50%, or walk-forward Sharpe ≤ 0, or the in-sample winner loses out of sample in ≥ 50% of splits |
| ACQUITTED | P(real) ≥ 95%, walk-forward Sharpe > 0, winner loses out of sample in < 20% of splits, look-ahead risk not high |
| HUNG JURY | anything in between |
| Sealed evidence | CONFIRMED if P(real) ≥ 95% on the sealed data (corrected for repeat openings); REFUTED if its Sharpe ≤ 0; else INCONCLUSIVE |

## Sandboxes as an epistemic guarantee

```
python:3.12-slim ──> + numpy, pandas                 (cached)
                        ├──> + OPEN data + strategy   (research checkpoint)
                        │       ├──> branch: configs 1-8
                        │       ├──> branch: configs 9-16   ... in parallel microVMs
                        │       └──> ...
                        └──> + FULL data + ONE committed strategy   (evidence room, after commitment)
```

Isolation usually means "the agent can't break my machine". Here it also means "the agent can't have peeked": the sealed dates are not in any research checkpoint. A test proves it by recording every dataset handed to an executor. The executor is written against contree-sdk's real API (read from its source) and tested with a stand-in that executes the commands, producing returns identical to local execution.

## Model routing

| Role | Tier | Why this tier |
|---|---|---|
| Planner | Nemotron 3 Ultra | Turning a vague idea into a falsifiable spec needs real reasoning |
| Engineer | Nemotron 3 Super | Strong code generation at a fraction of Ultra's price |
| Fisherman | Nemotron 3 Super | Decides where to fish next from the history of attempts |
| Auditor | Nemotron 3 Nano | Short focused review on every code revision |
| Clerk | Nemotron 3 Nano | Classifying search snippets is cheap work |
| Judge | Nemotron 3 Ultra | Must explain statistics faithfully without inventing numbers |

Every call is logged with tokens, latency and cost.

<!-- LIVE NUMBERS: replace this line with the output of a live trial and expedition: cost per trial, cost per expedition, sandbox wall time vs local -->

## Quickstart

```bash
git clone https://github.com/ekjots1ngh/overfit-court && cd overfit-court
pip install -r requirements.txt
cp .env.example .env                 # NEBIUS_API_KEY, NEBIUS_PROJECT_ID, TAVILY_API_KEY
python -m overfit_court.preflight    # checks models, sandboxes and Tavily end to end
streamlit run app.py
```

Without keys the app runs in demo mode (canned model responses, built-in strategy library, local execution), so anyone can try it immediately.

CLI:

```bash
python -m overfit_court "buy last quarter's winners, short the losers" --csv prices.csv --unseal
python -m overfit_court fish --csv prices.csv --target 1.0 --attempts 40
```

Real data: `pip install yfinance && python examples/fetch_prices.py sectors`.

Tests: `pytest -q` (24 tests, including sealed-evidence isolation, Reality Check calibration and duplicate invariance, distinct-bets counting, the court's memory, fishing expeditions, the sandbox executor against a fake contree client, and every strategy family through the harness).

Deploy: the `Dockerfile` serves the app on port 8501 (for example on Nebius Serverless Endpoints); CPU is enough.

## Nebius and NVIDIA usage

- **Nebius Token Factory**: every model call, through the OpenAI-compatible API (`overfit_court/llm.py`).
- **NVIDIA Nemotron 3 Nano, Super and Ultra**: routed by role as above.
- **Token Factory Sandboxes** (contree-sdk): checkpointed, branched execution of model-written code, plus the evidence room (`overfit_court/sandbox.py`).
- **Tavily**: runtime prior-art search (`overfit_court/research.py`).

## Project layout

```
overfit_court/
  stats.py      Reality Check, DSR/PSR, CSCV, walk-forward, min track record, verdict
  rmt.py        correlation spectrum, Marchenko-Pastur band, distinct bets
  ledger.py     the court record: every configuration's returns, every seal opening
  pipeline.py   trials, sealed evidence split, evidence room, commitments
  fishing.py    the fishing expedition, scripted and Nemotron fishermen
  library.py    8 strategy families and the universe-selection degree of freedom
  agents.py     planner, engineer, auditor, clerk, judge
  llm.py        Token Factory client, tier routing, cost ledger
  sandbox.py    harness: Token Factory Sandboxes, local subprocess, in-process (trusted only)
  research.py   Tavily prior-art search
  preflight.py  live setup check
app.py          Streamlit UI: trial, fishing expedition, how it works
examples/       fishing experiment, calibration, real data download
docs/           submission kit, feedback log
```

## Built during the submission period

All code was written after 26 August 2026 for this hackathon. The distinct-bets method is motivated by the author's earlier independent research on spurious factors in nonstationary markets (random matrix theory), but no code was reused.

## Feedback on Nebius and NVIDIA tools

See [docs/FEEDBACK.md](docs/FEEDBACK.md).

## Licence

MIT, see [LICENSE](LICENSE).
