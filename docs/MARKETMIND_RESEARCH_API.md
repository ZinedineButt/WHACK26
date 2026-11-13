# MarketMind Research API

Autonomous research engine. **Gemini proposes falsifiable hypotheses from a catalogue of real market and macro variables; Python tests every one** through correlation -> causal investigation -> heavy validation, and records everything, including nulls. All statistics come from executed code; Gemini never produces numbers and nothing it returns is executed.

Teammate handoff: fetch a finished report with `GET /api/research/runs/<id>` and plot it. No statistics, AI or data logic is needed client-side.

## Setup

```bash
cd backend
python -m venv .venv --system-site-packages        # or a plain venv
.venv/Scripts/pip install -r requirements.txt       # Linux/macOS: .venv/bin/pip
cp .env.example .env                                # then fill in keys (below)
.venv/Scripts/python app.py                         # http://localhost:5000  (PORT=... to change)
.venv/Scripts/python -m pytest tests -q             # 133 tests, ~110 s, fully offline
cd ../frontend && npm.cmd install && npm.cmd run dev   # "Research" tab  (npm run dev on macOS/Linux)
```

Research secrets (`GEMINI_*`, `FRED_API_KEY`) go in **`backend/.env.local`** (git-ignored by `.env.*`), so the tracked `backend/.env` used by the market map is never modified. Alpaca keys are shared with the map and read from `backend/.env`.

| Env var | Purpose |
|---|---|
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Required for `autonomous` / `guided`. Default model `gemini-3.5-flash-lite` (older `gemini-2.5-*` ids are retired for new accounts). |
| `FRED_API_KEY` | Optional. Without it the public FRED CSV endpoint is used (same series, no vintages). Free key: fred.stlouisfed.org/docs/api |
| `OPENALEX_API_KEY` | Optional but recommended. Free key from openalex.org; without it OpenAlex often answers HTTP 429 and the engine falls back to Semantic Scholar, then arXiv (keyless, finance/economics only, weaker relevance). |
| `TIINGO_API_KEY` | **Recommended for stocks/ETFs.** Free key from tiingo.com; end-of-day prices adjusted for splits AND dividends. Put it in `backend/.env.local`. The research engine does not need Alpaca at all. |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` (`ALPACA_SECRET_KEY` also accepted), `ALPACA_DATA_FEED` | Enables stock/ETF tickers (historical daily adjusted bars, default feed `iex`). Same keys as the live map. |
| `RESEARCH_JOB_TIMEOUT` (600 s), `RESEARCH_CACHE_TTL` (21600 s), `RESEARCH_SYNC` | Job timeout, provider cache lifetime, inline execution for tests. |

Keys are read server-side only and never appear in responses, logs or the frontend (there is a test for this).

## Modes

| `mode` | Live? | What happens |
|---|---|---|
| `autonomous` | yes | You give an optional theme / lookback / watchlist. Gemini picks the variables and hypotheses. |
| `guided` | yes | You give a question (+ optional series). Gemini translates it into hypotheses; unsupported questions fail clearly (`unsupported_reason`). |
| `demo_autonomous` | **no** | Scripted hypotheses on SYNTHETIC data with planted truths (a lagged effect, a null, a common-factor confound). Offline, never calls Gemini, labelled `template_demo` / `data_provenance: "synthetic"`. |
| legacy `{"preset_id": ...}` | no | Old synthetic presets (`demo_null`, `demo_confounder`, `demo_effect`), unchanged schema 1.0 report. |

Without `GEMINI_API_KEY`, live modes return **HTTP 503 "Gemini is not configured"**. They never silently fall back to canned hypotheses.

## Endpoints

| Method | URL | Notes |
|---|---|---|
| GET | `/api/health` | Host app's health (adds `gemini_configured`, `alpaca_configured`, `fred_mode` only if the host app has no `/api/health`). |
| GET | `/api/research/health` | `{status, gemini_configured, alpaca_configured, fred_mode}` |
| GET | `/api/research/catalog` | Providers actually configured now, every variable with `available` / `unavailable_reason`, allowed transforms, lookbacks `[1,3,5]`, hypothesis budget `1..5`, example themes, modes. |
| POST | `/api/research/runs` | **202** `{run_id, status:"queued", tracking_url}`. Errors: 400 invalid input, 413 too large, 429 more than 3 active runs, 503 Gemini missing. |
| GET | `/api/research/runs/<id>` | Full report (below), 404 if unknown. Poll every ~1 s. |
| GET | `/api/research/runs?limit=&offset=` | History, **including failed runs**. |
| POST | `/api/research/runs/<id>/simulate` | Paper-trade the frozen strategy of one completed hypothesis and compare it with holding the S&P 500. Body `{hypothesis_id, capital? (100..10,000,000, default 10,000), period? ("holdout" default | "development" | "full"), cost_bps? (0..50, default 2)}`. Errors: 400 invalid / outcome not tradable, 404 unknown run or hypothesis, 409 run unfinished or hypothesis did not complete the pipeline, 502 benchmark data unavailable. **Places no orders.** |
| GET | `/api/research/presets` | Legacy presets. |

Request bodies (fields beyond these are rejected):

```json
{"mode":"autonomous","theme":"US equities, bond yields and volatility","lookback_years":3,"max_hypotheses":3,
 "universe":{"tickers":["SPY","QQQ","JETS"],"macro_series":["DGS10","VIXCLS","DCOILWTICO"]}}
{"mode":"guided","question":"Do rising Treasury yields predict lower technology-stock returns?","lookback_years":3,
 "universe":{"tickers":["QQQ"],"macro_series":["DGS10"]}}
{"mode":"demo_autonomous"}
```
`universe` is optional (omit it and Gemini chooses from the whole catalogue). `max_hypotheses` is hard-capped at 5. Tickers must match `^[A-Z]{1,5}([.-][A-Z])?$` and be fetchable; macro ids must be in the allowlist.

```bash
curl localhost:5000/api/research/catalog
curl -XPOST localhost:5000/api/research/runs -H "Content-Type: application/json" \
  -d '{"mode":"autonomous","theme":"US equities, bond yields and volatility","lookback_years":3,"max_hypotheses":3}'
curl localhost:5000/api/research/runs/<run_id>
curl -XPOST localhost:5000/api/research/runs -H "Content-Type: application/json" -d '{"mode":"demo_autonomous"}'
# errors:
# {"error":"Gemini is not configured: set GEMINI_API_KEY in backend/.env to use live 'autonomous' mode, or use mode 'demo_autonomous' (offline, synthetic)."}   (503)
# {"error":"macro series 'CPIAUCSL' rejected: Monthly series needs release-date-aware (ALFRED vintage) data, which is not implemented; ..."}                 (400)
# {"error":"ticker 'SPY' rejected: Alpaca market-data credentials are not configured, so stock/ETF tickers cannot be fetched ..."}                          (400)
```

## Data sources and conventions

* **FRED** (daily): `DGS10`, `DGS2`, `DFF`, `T10Y2Y`, `BAMLH0A0HYM2`, `VIXCLS`, `DCOILWTICO`, `DCOILBRENTEU`, `DTWEXBGS`, and equity indices `SP500`, `NASDAQCOM`, `DJIA`. **Alpaca**: any valid ticker (adjusted daily close, `adjustment=all`, feed recorded in provenance). Providers sit behind adapters (`research/providers.py`) with timeouts, retry/backoff (429/5xx), pagination and a sqlite response cache (provenance records `retrieved_at`, `from_cache`, `first_date`, `last_date`).
* **Monthly macro (`CPIAUCSL`, `UNRATE`) is catalogued but UNSUPPORTED** for predictive/causal tests: no release-date-aware (ALFRED vintage) data is implemented, and the engine refuses rather than pretend the month's final value was known on day 1.
* **Timing**: an observation dated `t` is a market close known at `t` (FRED publication delay is not modelled for market-based series). Predictor at day `t` vs outcome strictly later. **Levels are inner-joined on common dates before any transform**, so no transform mixes mismatched calendars.
* **Transforms** (allowlisted per variable kind): `log_return` (prices, %), `diff` (rates/spreads, percentage points), `log_diff` and `level` (VIX-type indices; `level` also for spreads), `fwd_logvol_5|10` (outcome only: log annualised realised volatility of returns over t+1..t+h), plus server-added controls. Raw trending price levels are never correlated.

## Run lifecycle

`status`: `queued` -> `running` -> `completed` | `failed`. `current_phase` (+ `current_hypothesis`): `queued` -> `hypothesis_generation` -> `preregistered` -> then **per hypothesis** `correlation` -> `causality` -> `validation` -> `backtest` -> `generalisation` -> `family_correction` -> `critique` -> `complete`. Everything is saved after each step, so polling returns partial results. A hypothesis whose data cannot be fetched is `rejected` (kept, with reasons) and the rest continue; a hypothesis that crashes is `failed` and the rest continue; no testable hypothesis at all fails the run with `errors[].code` in `gemini_error | no_valid_hypotheses | no_testable_hypotheses | timeout | internal_error | interrupted`.

Execution is a bounded `ThreadPoolExecutor(2)` inside the Flask process (single process only; max 3 active runs). Timeout `RESEARCH_JOB_TIMEOUT`, checked between steps. Runs left unfinished by a restart are marked `failed/interrupted` on next start. Typical timings (measured, this machine): demo ~15 s; live guided, 1 hypothesis ~10 s; live autonomous, 3 hypotheses ~45 s (Gemini calls and literature retrieval dominate; typical live run of 3 hypotheses with backtest, peers and critique ~40-90 s; transient Gemini 503/429 are retried with backoff).

## Method (thresholds in `backend/research/criteria.py`, fixed before any run)

0. **Hypothesis generation, literature viability & pre-registration.** Before asking Gemini, related papers on the topic are retrieved from OpenAlex and given to it as background, with the instruction to propose only viable, realistic relationships (not known nulls, mechanical or arbitraged-away ones). Gemini **over-generates** (2x the budget, max 8 candidates). Candidates are then (a) validated, (b) screened for **data availability first**, so a candidate whose data cannot be fetched never consumes the budget, then (c) each gets its own literature search and Gemini scores its **viability 1-5** and writes a short summary of what the related research suggests (`literature_prior`: `viability`, `expectation` supports/mixed/contradicts/insufficient, `summary`, `cited_papers` by retrieved index only). Viability 1 is rejected as unrealistic; the top `max_hypotheses` by viability are kept; the rest are recorded as rejected with the reason. Only the selected hypotheses are frozen in the pre-registration.
   The remainder of phase 0: Gemini sees the live catalogue (never fixtures) and must return structured hypotheses (`predictor`, `outcome`, transforms, `direction`, `lag_days`, `null_hypothesis`, `mechanism`, `competing_explanations`, `control_variables`, `proposed_test`, `limitations`). The server validates each against the registry (variable exists/available, transform allowed, predictor leads, only `lagged_correlation_hac` test, <=3 controls); anything else is `rejected` with reasons and kept in `hypotheses`. Over-budget proposals are recorded too. After data screening and **before any statistic is computed**, the plan is frozen (`preregistration.sha256`). Every sample is split chronologically: stages 1-2 see only the first 75%; the last 25% is the holdout.
1. **Correlation** (development sample). **If no correlation is detected (`not_detected`), the hypothesis stops here**: status `no_correlation`, stages 2-6 are `skipped` ("Stopped: no correlation"), no causal/validation/backtest/peer work is done and no extra AI calls are made. They are listed under the heading **"Initial hypotheses with no correlation"** (`summary.no_correlation_label`, `summary.no_correlation[]`). Their correlation graph and related-research summary are still available, and their p-values still count in the run-level multiple-testing family. Details of the stage: Pearson r at the pre-registered lag, Fisher CI, iid p, **HAC (Newey-West) p** (drives the verdict), Spearman, effect size, ADF warnings, same-time r (descriptive), and the pre-registered lag family 1..5 Holm-corrected. `detected` = HAC p<0.01, |r|>=0.10, Spearman same sign, sign matches the pre-registered direction; `not_detected` = HAC p>=0.05; else `inconclusive`.
2. **Causality.** Gemini proposes mechanisms, alternatives, confounders and a DAG (nodes restricted to catalogue ids or `latent:*`; invalid nodes dropped). Python runs: prewhitened lead/lag scan (Holm), regression control for observed common factors dated at the predictor time (pre-outcome, to avoid post-treatment bias), Granger both ways, circular-shift placebo. `negative_control` and `event_study` are always reported as `not_run` with a reason (no inputs exist), never fabricated; forward-volatility outcomes also `not_run` lead/lag + Granger. `causal_verdict`: `contradicted` (controls remove >=50% of the effect or only outcome-leads-predictor), `evidence_consistent` (association detected and lead/lag, control, Granger and placebo all pass), else `not_established`. **Never `proven`.**
3. **Validation.** 50/25/25 train/validation/holdout. Robustness diagnostics (replication, HAC+Holm over lags tried, block bootstrap, best-of-lags placebo, 4-window stability, winsorisation, sign consistency) use ONLY train+validation. The **holdout is evaluated by exactly one test** (`holdout_single_evaluation`, `holdout_evaluations: 1`; tests prove that corrupting the holdout changes nothing else). Then a run-level `family_multiple_testing` test applies **Holm across every hypothesis actually tested in the run** (on top of each hypothesis's own lag-family Holm). `reliability_verdict`: `rejected` (holdout fails or >=2 critical fail), `supported` (all critical pass, <=1 other fails), else `inconclusive`. Test outcomes: `pass | fail | inconclusive | not_run`.

4. **Backtest (`hypothesis.backtest`).** The hypothesis is turned into a rule fixed before any statistic: hold `+/- sign(predictor_t)` from the close of day `t`, earn the outcome return of day `t+lag`, pay 2 bps per unit of turnover. The sign is the pre-registered direction (the development-sample sign if `any`). Reported separately for the development sample and the holdout: net Sharpe, hit rate (directional accuracy), total return vs buy-and-hold, max drawdown, turnover, cost drag, and a block-bootstrap CI of the mean daily net return. `pass` = holdout net mean > 0 AND CI above 0 AND hit rate > 50%; `fail` = net mean <= 0; else `inconclusive`. Only for return outcomes (`applicable: false` for volatility outcomes). The holdout P&L restates the same frozen rule; nothing is tuned on it. It does not change `reliability_verdict`.
5. **Peer generalisation (`hypothesis.generalisation`).** If, and only if, the primary association is detected, the same frozen hypothesis (predictor, transforms, lag, sign) is re-run on peer assets from a fixed **server-side peer map** (`research/peers.py`; Gemini cannot choose peers). Peers are listed in the pre-registration. Per peer: r, HAC p, Holm over peers, sign agreement, frozen-rule Sharpe and hit rate. `generalises` = >=60% of peers share the sign AND >=50% are significant that way after Holm; `partial`; `does_not_generalise`; `not_run`.
6. **Literature and self-critique (`hypothesis.literature`, `hypothesis.critique`).** Real papers are retrieved from OpenAlex (set `OPENALEX_API_KEY`), falling back to Semantic Scholar and then arXiv; the source used is in `literature.provider`. Gemini then red-teams the finding against the computed results and the retrieved abstracts. Papers are referenced by index and joined server-side, so citations cannot be invented; critique points must be grounded in a real test id, paper, backtest, peers, causality or multiple_testing, otherwise they are dropped; unsupported numbers or the word "proven" trigger a labelled template fallback; `confidence` is capped server-side by the evidence (never `high` unless supported + backtest pass + generalises + evidence_consistent). Retrieval failure degrades to `literature.status: "unavailable"` without affecting statistics. Literature informs the critique only; it never changes a verdict.

"Supported" means a reliably replicating association after multiple-testing correction. It is **not** causal evidence. All findings are labelled exploratory.

## Report schema (`schema_version` "2.0"; keys stable in every status)

* Top level: `run_id` (= `id`), `status`, `mode`, `request` (normalised), `current_phase`, `current_hypothesis`, `phases[]`, `generation` (`source`, `model`, `requested_budget`, `proposals_received`, `unsupported_reason`, `catalog`), `preregistration`, `hypotheses[]`, `multiple_testing` (`proposed`, `attempted`, `correction`, `primary_p_hac`, `primary_p_holm`, `significant_after_correction`, ...), `visualization_data{<hypothesis_id>: {...}}`, `summary{overall, counts, hypotheses[]}`, `reasoning_source` (generation source: `gemini` | `template_demo`), `reasoning_sources{generation, causal[], narrative[]}` (each `gemini` | `template_demo` | `template_fallback`), `data_provenance` (`real` | `synthetic`), `warnings[]`, `errors[]`, `timestamps{created,started,finished,updated}`.
* Each `hypotheses[i]`: `hypothesis_id`, `status` (`pending|running|completed|failed|rejected`), `proposal` (raw model output), `normalized` (validated: `question`, `predictor{variable_id,label,transform}`, `outcome{...}`, `direction`, `lag_days`, `null_hypothesis`, `mechanism`, `competing_explanations`, `control_variables`, `market_control`, `required_datasets`, `limitations`), `eligibility{accepted,reasons,notes}`, `stages[{id,status,message}]`, `provenance` (datasets with source/feed/`retrieved_at`, transformations, timing, sample dates, `development_n`, `holdout_n`), `correlation`, `causality`, `validation`, `verdict{association,causal,reliability,evidence_level,holdout_confirmation,statement}`, `summary`, `warnings`, `error`.
* `hypotheses[i].status` may also be `no_correlation` (completed stage 1, stopped). `literature_prior{viability, expectation, summary, concerns, cited_papers[], source, papers_retrieved, search_status}` appears for live modes. `generation` adds `candidate_pool` and `background_literature`; `multiple_testing` adds `candidates_after_validation` and `candidates_with_data`; `reasoning_sources.ranking`; `summary.counts.no_correlation`.
* Added per hypothesis: `backtest{applicable, rule, sign, sign_source, cost_bps, development{...}, holdout{n, sharpe, hit_rate, total_return_pct, buy_hold_total_return_pct, max_drawdown_pct, mean_daily_net_ci_95, ...}, verdict, verdict_rule, caveats}`, `generalisation{status, verdict, peers[{variable_id, label, n, r, p_hac, p_holm, same_sign_as_primary, significant_after_holm, backtest{sharpe,hit_rate,...}}], skipped[], share_same_sign, share_significant_same_sign, verdict_rule}`, `literature{status, query, provider, papers[{index,title,year,doi,url,venue,cited_by,abstract}]}`, `critique{paper_assessments[{paper_index,relevance,stance,note}], critique[{point,severity,grounded_in}], final_assessment, confidence, confidence_cap, source}`; `verdict` gains `backtest`, `generalisation`, `confidence`. Stage ids: `correlation, causality, validation, backtest, generalisation, critique` (`skipped` = not applicable, with a message).
* `correlation`: `r`, `p_value`, `p_value_hac`, `ci_95`, `n`, `effect_size_r2`, `spearman_rho/p`, `same_time_r`, `primary_lag`, `lag_family[{lag,r,p_hac,p_holm}]`, `association_verdict`, `verdict_rule`, `assumptions`, `checks`, `warnings`.
* `causality`: `mechanisms`, `alternative_explanations`, `possible_confounders`, `assumptions`, `tests[]`, `causal_verdict`, `verdict_rule`, `verdict_criteria`, `reasoning`, `lag_profile`. `validation`: `tests[]` (each with `critical`), `reliability_verdict`, `verdict_rule`, `claim_tested`, `split`, `holdout_evaluations`, `reasoning`. Every test: `{id, name, metrics, threshold, result, explanation, limitations}`.
* `visualization_data[H]`: `backtest_equity[{date, strategy, buy_hold, strategy_holdout, buy_hold_holdout, drawdown, position, sample}]` (`*_holdout` = rebased cumulative % from the first holdout day, null before), `peers` (same rows as `generalisation.peers`), `scatter_fit{slope, intercept, x_min, x_max}` (drawing only), `timeseries[{date,predictor,outcome,c_<control>...,sample}]`, `scatter[{x,y,date,sample}]` (predictor at t vs outcome at t+lag; `sample` = `development`|`holdout`), `rolling_r[{date,r}]` (null until the window is full), `lag_profile`, `validation_metrics{window_r, r_by_split, bootstrap_ci_95, split}`, `causal_graph{nodes[{id,label,role,observed}], edges[{source,target,status,kind,evidence}]}`. Edge `status`: `consistent_with_evidence|contradicted|not_established|supported_by_test|not_supported|hypothesised|proposed_untested`.
* JSON is strict (no NaN/Infinity/numpy; non-finite values become `null`).

## Real example

Live `autonomous` run (Gemini `gemini-3.5-flash-lite` + FRED public CSV, 2026-11-07, 3-year lookback, no tickers named by the user). Only hypothesis H2 is shown, arrays truncated; the run's other two hypotheses were null results (r=0.046, p_HAC=0.405; r=0.006, p_HAC=0.916). Summary: **0 supported, 3 rejected**.

```json
{
  "current_hypothesis": null,
  "current_phase": "complete",
  "data_provenance": "real",
  "errors": [],
  "generation": {
    "catalog": [
      "DGS10",
      "DGS2",
      "DFF",
      "T10Y2Y",
      "..."
    ],
    "model": "gemini-3.5-flash-lite",
    "proposals_received": 3,
    "requested_budget": 3,
    "source": "gemini",
    "unsupported_reason": ""
  },
  "hypotheses": [
    {
      "causality": {
        "alternative_explanations": [
          "Reverse causation where actual early realized shocks instantaneously re-price the options chain before the 5-day window is fully computed",
          "Shared global liquidity and monetary policy factors driving both short-term bond yields and equity uncertainty simultaneously",
          "Coincident macroeconomic news announcements that independently shock both the VIX and the S&P 500 index variance on the same day"
        ],
        "assumptions": [
          "The 5-day forward window on the S&P 500 effectively prevents contemporaneous feedback from entering the outcome variable.",
          "Unobserved global macro shocks represent the primary omitted common driver between implied and realized volatility."
        ],
        "causal_verdict": "not_established",
        "lag_profile": [],
        "mechanisms": [
          "An increase in options-implied volatility reflects rising market uncertainty and hedging demand, which translates into elevated realized price fluctuations over the subsequent trading week as market makers rebalance portfolios."
        ],
        "possible_confounders": [
          {
            "rationale": "Fluctuations in the trade-weighted U.S. dollar index can simultaneously drive equity market sentiment and safe-haven demand, confounding the relationship between VIX and future equity volatility.",
            "variable_id": "DTWEXBGS"
          },
          {
            "rationale": "Changes in 2-year Treasury yields capture shifting monetary policy expectations and discount rates, which jointly influence equity volatility and option pricing.",
            "variable_id": "DGS2"
          }
        ],
        "reasoning": "Regarding causality, the analysis between the CBOE VIX log difference and the S&P 500 index forward log volatility yielded a correlation coefficient of 0.16698920707785855 across 548 observations, where an association was detected. However, a causal reading was not established because not all required checks passed or could be run, meaning we cannot state that changes in the VIX directly cause shifts in future index volatility.",
        "reasoning_source": "gemini",
        "tests": [
          {
            "explanation": "Forward-window outcome: the outcome already starts after the predictor date, so reverse-direction scans are not meaningful.",
            "id": "lead_lag",
            "limitations": [],
            "metrics": {},
            "name": "Lead/lag scan",
            "result": "not_run",
            "threshold": "n/a"
          },
          {
            "explanation": "Predictor coefficient survives the controls (attenuation -13%, p=0.000199).",
            "id": "confounder_control",
            "limitations": [
              "Only observed controls; unobserved confounders remain possible.",
              "Controls are dated at the predictor time to avoid conditioning on post-treatment variables."
            ],
            "metrics": {
              "attenuation": -0.12899902611904634,
              "beta_controlled": 0.01279985000842445,
              "beta_raw": 0.011337343710936719,
              "controls": [
                "c_DGS2",
                "c_DTWEXBGS",
                "c_trailing_vol"
              ],
              "lag": 0,
              "n": 548,
              "p_controlled": 0.00019916155894994582,
              "p_raw": 0.002970755405315918
            },
            "name": "Regression control for observed common factors",
            "result": "pass",
            "threshold": "controlled p<0.05 and attenuation<50%"
          },
          {
            "explanation": "Forward-window outcome: the outcome already starts after the predictor date, so reverse-direction scans are not meaningful.",
            "id": "granger",
            "limitations": [],
            "metrics": {},
            "name": "Granger-style predictive test",
            "result": "not_run",
            "threshold": "n/a"
          },
          {
            "explanation": "Association is stronger than for time-shifted copies of X (not a trend/autocorrelation artefact).",
            "id": "placebo_shift",
            "limitations": [
              "Rules out spurious co-trending only. A shared factor still passes this test."
            ],
            "metrics": {
              "lag": 0,
              "p_placebo": 0.002
            },
            "name": "Circular-shift placebo (negative control)",
            "result": "pass",
            "threshold": "p<0.05"
          },
          {
            "explanation": "No pre-specified negative-control outcome exists in the catalogue.",
            "id": "negative_control",
            "limitations": [],
            "metrics": {},
            "name": "Negative-control outcome",
            "result": "not_run",
            "threshold": "n/a"
          },
          {
            "explanation": "No event/release-date calendar is available.",
            "id": "event_study",
            "limitations": [],
            "metrics": {},
            "name": "Event study",
            "result": "not_run",
            "threshold": "n/a"
          }
        ],
        "verdict_criteria": {
          "contradicted": "association detected AND (control regression fails OR only outcome-leads-predictor lags significant)",
          "evidence_consistent": "association detected AND lead_lag, confounder_control, granger and placebo_shift all pass",
          "not_established": "everything else (default). 'proven' is never used."
        },
        "verdict_rule": "Not all required checks passed or could be run; causal reading not established."
      },
      "correlation": {
        "association_verdict": "detected",
        "assumptions": [
          "Pearson r is linear association only.",
          "p_value assumes i.i.d.; p_value_hac (Newey-West) drives the verdict.",
          "Only pre-registered lags 1..5 were tested; Holm-corrected across them."
        ],
        "checks": {
          "adf_p_x": 5.0070255592223353e-23,
          "adf_p_y": 1.4281553087387043e-05
        },
        "ci_95": [
          0.08441091652019687,
          0.2472893435126207
        ],
        "criteria": {
          "detected": "HAC p<0.01, |r|>=0.1, Spearman same sign, sign matches direction",
          "not_detected": "HAC p>=0.05"
        },
        "direction_consistent": true,
        "direction_preregistered": "positive",
        "effect_size_r2": 0.027885395280491925,
        "hac_maxlags": 10,
        "lag_family": [
          {
            "lag": 0,
            "p_hac": 0.0029707554053158805,
            "p_holm": 0.0029707554053158805,
            "r": 0.16698920707785853
          }
        ],
        "lag_family_correction": "holm",
        "method": "pearson",
        "n": 548,
        "p_value": 8.572525824780051e-05,
        "p_value_hac": 0.0029707554053158805,
        "primary_lag": 0,
        "r": 0.16698920707785855,
        "same_time_r": 0.16698920707785858,
        "sample": {
          "end": "2026-01-30",
          "label": "development sample (first 75%); holdout sealed",
          "n_rows": 548,
          "start": "2023-11-16"
        },
        "spearman_p": 0.0004078193190097376,
        "spearman_rho": 0.1504893133734675,
        "timing": "predictor stamped t vs outcome over t+1..t+5 (forward window)",
        "verdict_rule": "HAC p=0.00297 < 0.01, |r|=0.167 >= 0.1, Spearman agrees, sign matches pre-registered direction.",
        "warnings": [
          "Forward-window outcome: observations overlap, so HAC lags were widened to 10."
        ],
        "x_label": "CBOE VIX [log_diff]",
        "y_label": "S&P 500 index [fwd_logvol_5]"
      },
      "eligibility": {
        "accepted": true,
        "data_checked": true,
        "notes": [],
        "reasons": []
      },
      "error": null,
      "hypothesis_id": "H2",
      "normalized": {
        "competing_explanations": [
          "VIX is merely a contemporaneous reflection of current equity returns rather than a forward-looking predictor.",
          "Market maker inventory adjustments alter implied volatility independently of future realized risk."
        ],
        "control_variables": [
          "DGS2",
          "DTWEXBGS"
        ],
        "direction": "positive",
        "lag_days": 0,
        "limitations": [
          "Implied volatility can incorporate risk premiums that do not materialize as realized volatility.",
          "Overlapping 5-day rolling windows induce autocorrelation in the outcome variable."
        ],
        "market_control": [],
        "mechanism": "An increase in options-implied volatility reflects rising market uncertainty and hedging demand, which translates into elevated realized price fluctuations over the subsequent trading week.",
        "null_hypothesis": "Changes in the CBOE VIX do not predict the subsequent 5-day realized volatility of the S&P 500.",
        "outcome": {
          "label": "S&P 500 index",
          "transform": "fwd_logvol_5",
          "variable_id": "SP500"
        },
        "predictor": {
          "label": "CBOE VIX",
          "transform": "log_diff",
          "variable_id": "VIXCLS"
        },
        "proposed_test": "lagged_correlation_hac",
        "question": "Does the daily log difference of the CBOE VIX predict future 5-day realized volatility of the S&P 500?",
        "required_datasets": [
          "DGS2",
          "DTWEXBGS",
          "SP500",
          "VIXCLS"
        ]
      },
      "proposal": {
        "competing_explanations": [
          "VIX is merely a contemporaneous reflection of current equity returns rather than a forward-looking predictor.",
          "Market maker inventory adjustments alter implied volatility independently of future realized risk."
        ],
        "control_variables": [
          "DGS2",
          "DTWEXBGS"
        ],
        "direction": "positive",
        "hypothesis_id": "H2",
        "lag_days": 0,
        "limitations": [
          "Implied volatility can incorporate risk premiums that do not materialize as realized volatility.",
          "Overlapping 5-day rolling windows induce autocorrelation in the outcome variable."
        ],
        "mechanism": "An increase in options-implied volatility reflects rising market uncertainty and hedging demand, which translates into elevated realized price fluctuations over the subsequent trading week.",
        "null_hypothesis": "Changes in the CBOE VIX do not predict the subsequent 5-day realized volatility of the S&P 500.",
        "outcome_transform": "fwd_logvol_5",
        "outcome_variable": "SP500",
        "predictor_transform": "log_diff",
        "predictor_variable": "VIXCLS",
        "proposed_test": "lagged_correlation_hac",
        "question": "Does the daily log difference of the CBOE VIX predict future 5-day realized volatility of the S&P 500?"
      },
      "provenance": {
        "alignment": "levels inner-joined on common observation dates BEFORE transforming; no look-ahead",
        "data_type": "real",
        "development_n": 548,
        "holdout_n": 183,
        "observations": 731,
        "raw_rows_aligned": 741,
        "rows_lost_to_alignment_or_transform": 10,
        "sample_end": "2026-10-23",
        "sample_start": "2023-11-16",
        "timing": "predictor observed at the close of day t; outcome measured strictly after t",
        "transformations": {
          "controls": {
            "DGS2": "diff",
            "DTWEXBGS": "log_return"
          },
          "outcome": "fwd_logvol_5",
          "predictor": "log_diff"
        },
        "usable_rows": 731,
        "variables": [
          {
            "first_date": "2023-11-08",
            "from_cache": false,
            "label": "2-year Treasury yield",
            "last_date": "2026-11-05",
            "note": "latest-revised values; point-in-time vintages not used (valid for market-based daily series)",
            "observations": 748,
            "provider": "fred_public_csv",
            "requested_end": "2026-11-07",
            "requested_start": "2023-11-08",
            "retrieved_at": "2026-11-07T16:04:44.758440+00:00",
            "series_id": "DGS2",
            "units": "percent",
            "variable_id": "DGS2"
          },
          "... (one entry per dataset)"
        ]
      },
      "stages": [
        {
          "id": "correlation",
          "message": "Computed association statistics",
          "status": "completed"
        },
        {
          "id": "causality",
          "message": "Ran causal checks",
          "status": "completed"
        },
        {
          "id": "validation",
          "message": "Finished validation battery",
          "status": "completed"
        }
      ],
      "status": "completed",
      "summary": "r=0.167 at lag 0 (n=548, HAC p=0.00297); Association detected; causation not established; reliability rejected. Association alone never establishes causation.",
      "validation": {
        "claim_tested": "X lagged by 0 day(s) (pre-registered lag) is linearly associated with Y.",
        "holdout_evaluations": 1,
        "reasoning": "In terms of validation, the reliability verdict was rejected because it failed the holdout single evaluation test, even though it passed multiple other checks including validation replication, HAC Holm, block bootstrap, placebo max statistic, window stability, outlier sensitivity, direction consistency, and family multiple testing. Because of this single failure in the holdout evaluation, the overall reliability remains unverified.",
        "reliability_verdict": "rejected",
        "selected_lag": 0,
        "split": {
          "holdout": [
            "2026-02-02",
            "2026-10-23",
            183
          ],
          "train": [
            "2023-11-16",
            "2025-05-06",
            365
          ],
          "validation": [
            "2025-05-07",
            "2026-01-30",
            183
          ]
        },
        "tests": [
          {
            "critical": true,
            "explanation": "Train-selected relationship replicates on later data.",
            "id": "validation_replication",
            "limitations": [
              "Single split; a short validation window has limited power."
            ],
            "metrics": {
              "lag": 0,
              "n": 183,
              "p_hac_validation": 0.0005802045107450379,
              "r_train": 0.16862925109687887,
              "r_validation": 0.16027201967359933
            },
            "name": "Replication on validation split (lag frozen from train)",
            "result": "pass",
            "threshold": "HAC p<0.05 and same sign as train"
          },
          {
            "critical": true,
            "explanation": "Frozen relationship fails on the untouched final data.",
            "id": "holdout_single_evaluation",
            "limitations": [
              "Holdout is used once; repeated looking would invalidate it.",
              "OOS R2 for lag 0 is explanatory, not forecasting."
            ],
            "metrics": {
              "lag": 0,
              "n": 183,
              "oos_r2": -0.019715528876333543,
              "p_hac_holdout": 0.13180091315316525,
              "r_holdout": 0.06570540944836432
            },
            "name": "Final holdout (evaluated once)",
            "result": "fail",
            "threshold": "HAC p<0.05, same sign, out-of-sample R2>0"
          },
          {
            "critical": true,
            "explanation": "Survives Newey-West errors and the lag-selection multiplicity correction.",
            "id": "hac_holm",
            "limitations": [
              "Holm controls family-wise error only across the lags attempted here."
            ],
            "metrics": {
              "lag": 0,
              "lags_tried": [
                0
              ],
              "n_tests_attempted": 1,
              "p_hac": 0.0029707554053158805,
              "p_holm": 0.0029707554053158805
            },
            "name": "Full-sample HAC inference, Holm-corrected over lags tried",
            "result": "pass",
            "threshold": "Holm p<0.05"
          },
          {
            "critical": true,
            "explanation": "Block-bootstrap interval excludes zero.",
            "id": "block_bootstrap",
            "limitations": [
              "Block length is a fixed choice; very persistent series may need longer blocks."
            ],
            "metrics": {
              "block": 20,
              "ci_95": [
                0.08936697536461453,
                0.24067000746438894
              ],
              "draws": 1000,
              "r": 0.16698920707785858
            },
            "name": "Moving-block bootstrap 95% CI",
            "result": "pass",
            "threshold": "CI excludes 0"
          },
          {
            "critical": true,
            "explanation": "Stronger than the best lag of time-shifted placebo copies.",
            "id": "placebo_max_stat",
            "limitations": [
              "Preserves X's autocorrelation but breaks its alignment with Y."
            ],
            "metrics": {
              "draws": 499,
              "max_abs_r": 0.16698920707785858,
              "p_placebo": 0.002
            },
            "name": "Circular-shift placebo on best-of-lags statistic",
            "result": "pass",
            "threshold": "p<0.05"
          },
          {
            "critical": false,
            "explanation": "Effect direction is consistent across time.",
            "id": "window_stability",
            "limitations": [
              "Each window has only ~137 points."
            ],
            "metrics": {
              "same_sign": 4,
              "window_r": [
                0.17877451218767237,
                0.09997034286986964,
                0.2786393540892367,
                0.15356939341852874
              ],
              "windows": 4
            },
            "name": "Stability across chronological windows",
            "result": "pass",
            "threshold": ">=3/4 windows share the full-sample sign"
          },
          {
            "critical": false,
            "explanation": "Result is not driven by extreme days.",
            "id": "outlier_sensitivity",
            "limitations": [
              "Winsorising removes genuine tail events too."
            ],
            "metrics": {
              "p_hac_winsorised": 0.00017129744896892955,
              "r": 0.16698920707785858,
              "r_winsorised": 0.176653976357191,
              "relative_change": 0.057876610401690315
            },
            "name": "Winsorised (1%/99%) re-estimate",
            "result": "pass",
            "threshold": "change<=25% and still significant"
          },
          {
            "critical": false,
            "explanation": "Sign agrees in all splits.",
            "id": "direction_consistency",
            "limitations": [
              "Sign agreement is weak evidence on its own."
            ],
            "metrics": {
              "r_holdout": 0.06570540944836432,
              "r_train": 0.16862925109687887,
              "r_validation": 0.16027201967359933
            },
            "name": "Effect direction across train/validation/holdout",
            "result": "pass",
            "threshold": "all three share the same sign"
          },
          {
            "critical": true,
            "explanation": "Survives correction for the whole hypothesis family.",
            "id": "family_multiple_testing",
            "limitations": [
              "Family = hypotheses generated in this run; earlier runs are not counted."
            ],
            "metrics": {
              "family_size": 3,
              "p_hac": 0.0029707554053158805,
              "p_holm": 0.008912266215947642
            },
            "name": "Run-level Holm correction across all hypotheses tested",
            "result": "pass",
            "threshold": "Holm p<0.05"
          }
        ],
        "verdict_criteria": {
          "critical_tests": [
            "validation_replication",
            "holdout_single_evaluation",
            "hac_holm",
            "block_bootstrap",
            "placebo_max_stat"
          ],
          "inconclusive": "otherwise, or n<200",
          "rejected": "holdout fails OR >=2 critical tests fail",
          "supported": "all critical tests pass AND <=1 non-critical fails"
        },
        "verdict_rule": "Failed: holdout_single_evaluation."
      },
      "verdict": {
        "association": "detected",
        "causal": "not_established",
        "evidence_level": "exploratory",
        "holdout_confirmation": "fail",
        "reliability": "rejected",
        "statement": "Association detected; causation not established; reliability rejected. Association alone never establishes causation."
      },
      "warnings": []
    }
  ],
  "id": "06ba53a7-deac-477d-8189-86a1e2a04af3",
  "mode": "autonomous",
  "multiple_testing": {
    "attempted": 3,
    "correction": "holm",
    "family_alpha": 0.05,
    "hypotheses_in_family": 3,
    "note": "Holm correction over the primary-lag HAC p-value of every hy...",
    "primary_p_hac": {
      "H1": 0.40487897372102144,
      "H2": 0.0029707554053158805,
      "H3": 0.9161912716788169
    },
    "primary_p_holm": {
      "H1": 0.8097579474420429,
      "H2": 0.008912266215947642,
      "H3": 0.9161912716788169
    },
    "proposed": 3,
    "significant_after_correction": [
      "H2"
    ],
    "within_hypothesis_lag_family": [
      1,
      2,
      3,
      4,
      5
    ]
  },
  "phases": [
    {
      "id": "hypothesis_generation",
      "message": "3 proposed, 3 passed validation",
      "status": "completed"
    }
  ],
  "preregistration": {
    "alpha": 0.05,
    "correction": "holm",
    "family_size": 3,
    "frozen_at": "2026-11-07T16:04:45.680504+00:00",
    "frozen_before_any_statistic": true,
    "holdout": "last 25% of each sample; evaluated exactly once, after all hypotheses and methods were frozen",
    "hypotheses": [
      {
        "hypothesis_id": "H2",
        "plan": {
          "controls": [
            "DGS2",
            "DTWEXBGS"
          ],
          "direction": "positive",
          "lag_family": [
            0
          ],
          "outcome": {
            "label": "S&P 500 index",
            "transform": "fwd_logvol_5",
            "variable_id": "SP500"
          },
          "predictor": {
            "label": "CBOE VIX",
            "transform": "log_diff",
            "variable_id": "VIXCLS"
          },
          "primary_lag": 0,
          "split": {
            "development": [
              "2023-11-16",
              "2026-01-30"
            ],
            "holdout": [
              "2026-02-02",
              "2026-10-23"
            ]
          },
          "tests": "lagged_correlation_hac -> causal checks -> validation battery"
        }
      }
    ],
    "label": "Stages 1-2 are exploratory (development sample). Only the holdout is confirmatory.",
    "sha256": "a4a5dbc798bd075ec1de9686358cf1a4881ba3e6181b09eac67a5fa00e5f9455"
  },
  "reasoning_source": "gemini",
  "reasoning_sources": {
    "causal": [
      {
        "hypothesis_id": "H1",
        "source": "gemini"
      },
      {
        "hypothesis_id": "H2",
        "source": "gemini"
      },
      {
        "hypothesis_id": "H3",
        "source": "gemini"
      }
    ],
    "generation": "gemini",
    "narrative": [
      {
        "hypothesis_id": "H1",
        "source": "gemini"
      },
      {
        "hypothesis_id": "H2",
        "source": "gemini"
      },
      {
        "hypothesis_id": "H3",
        "source": "gemini"
      }
    ]
  },
  "request": {
    "lookback_years": 3,
    "max_hypotheses": 3,
    "mode": "autonomous",
    "theme": "US equities, bond yields and volatility",
    "universe": {
      "macro_series": [],
      "tickers": []
    }
  },
  "run_id": "06ba53a7-deac-477d-8189-86a1e2a04af3",
  "schema_version": "2.0",
  "status": "completed",
  "summary": {
    "counts": {
      "failed": 0,
      "inconclusive": 0,
      "rejected": 3,
      "supported": 0
    },
    "hypotheses": [
      {
        "hypothesis_id": "H1",
        "question": "Does daily change in the 10-year Treasury yield predict next-day log returns of the S&P 500?",
        "text": "r=0.046 at lag 1 (n=554, HAC p=0.405); Association not_detected; causation not established; reliability rejected. Association alone never establishes causation."
      },
      {
        "hypothesis_id": "H2",
        "question": "Does the daily log difference of the CBOE VIX predict future 5-day realized volatility of the S&P 500?",
        "text": "r=0.167 at lag 0 (n=548, HAC p=0.00297); Association detected; causation not established; reliability rejected. Association alone never establishes causation."
      },
      {
        "hypothesis_id": "H3",
        "question": "Does the daily change in the 2-year Treasury yield predict next-day log returns of the NASDAQ Composite?",
        "text": "r=0.006 at lag 1 (n=557, HAC p=0.916); Association not_detected; causation not established; reliability rejected. Association alone never establishes causation."
      }
    ],
    "overall": "3 hypotheses proposed, 3 tested to completion: 0 supported, 3 rejected, 0 inconclusive. All results are reported, including nulls. 'Supported' means the association replicated on the single holdout and survived multiple-testing correction; it is not proof of causation. Findings are exploratory."
  },
  "timestamps": {
    "created": "2026-11-07T16:04:29.766027+00:00",
    "finished": "2026-11-07T16:05:14.610174+00:00",
    "started": "2026-11-07T16:04:29.778762+00:00",
    "updated": "2026-11-07T16:05:14.610195+00:00"
  },
  "visualization_data": {
    "H2": {
      "causal_graph": {
        "edges": [
          {
            "evidence": [
              "lead_lag",
              "confounder_control",
              "granger",
              "placebo_shift"
            ],
            "kind": "hypothesised_causal",
            "source": "VIXCLS",
            "status": "not_established",
            "target": "SP500"
          },
          {
            "evidence": [
              "confounder_control"
            ],
            "kind": "possible_confounder",
            "source": "DGS2",
            "status": "not_supported",
            "target": "VIXCLS"
          },
          {
            "evidence": [
              "confounder_control"
            ],
            "kind": "possible_confounder",
            "source": "DGS2",
            "status": "not_supported",
            "target": "SP500"
          },
          {
            "evidence": [
              "confounder_control"
            ],
            "kind": "possible_confounder",
            "source": "DTWEXBGS",
            "status": "not_supported",
            "target": "VIXCLS"
          },
          {
            "evidence": [
              "confounder_control"
            ],
            "kind": "possible_confounder",
            "source": "DTWEXBGS",
            "status": "not_supported",
            "target": "SP500"
          },
          {
            "evidence": [],
            "kind": "model_proposed",
            "rationale": "Unobserved macroeconomic shocks simultaneously drive implied volatility.",
            "source": "latent:macro_shock",
            "status": "proposed_untested",
            "target": "VIXCLS"
          },
          {
            "evidence": [],
            "kind": "model_proposed",
            "rationale": "Unobserved macroeconomic shocks simultaneously drive realized equity volatility.",
            "source": "latent:macro_shock",
            "status": "proposed_untested",
            "target": "SP500"
          }
        ],
        "nodes": [
          {
            "id": "VIXCLS",
            "label": "CBOE VIX",
            "observed": true,
            "role": "predictor"
          },
          {
            "id": "SP500",
            "label": "S&P 500 index",
            "observed": true,
            "role": "outcome"
          },
          {
            "id": "DGS2",
            "label": "DGS2",
            "observed": true,
            "role": "control"
          },
          {
            "id": "DTWEXBGS",
            "label": "DTWEXBGS",
            "observed": true,
            "role": "control"
          },
          {
            "id": "latent:macro_shock",
            "label": "latent:macro_shock",
            "observed": false,
            "role": "proposed"
          }
        ]
      },
      "lag_profile": [
        {
          "lag": 0,
          "p_hac": 0.0029707554053158805,
          "p_holm": 0.0029707554053158805,
          "r": 0.16698920707785853
        }
      ],
      "lead_lag_profile": [],
      "rolling_r": [
        {
          "date": "2023-11-16",
          "r": null
        },
        {
          "date": "2023-11-17",
          "r": null
        },
        "... 729 more items"
      ],
      "rolling_window": 60,
      "scatter": [
        {
          "date": "2023-11-16",
          "sample": "development",
          "x": 0.9824640428518272,
          "y": 1.7380659359198487
        },
        {
          "date": "2023-11-17",
          "sample": "development",
          "x": -3.6988569369340585,
          "y": 1.8638758079506514
        },
        "... 729 more items"
      ],
      "scatter_lag": 0,
      "timeseries": [
        {
          "c_DGS2": -0.07000000000000028,
          "c_DTWEXBGS": 0.08297203823124377,
          "c_trailing_vol": 2.858437455188702,
          "date": "2023-11-16",
          "outcome": 1.7380659359198487,
          "predictor": 0.9824640428518272,
          "sample": "development"
        },
        {
          "c_DGS2": 0.04999999999999982,
          "c_DTWEXBGS": -0.2708007626217146,
          "c_trailing_vol": 2.6115582120655896,
          "date": "2023-11-17",
          "outcome": 1.8638758079506514,
          "predictor": -3.6988569369340585,
          "sample": "development"
        },
        "... 729 more items"
      ],
      "validation_metrics": {
        "bootstrap_ci_95": [
          0.08936697536461453,
          0.24067000746438894
        ],
        "r_by_split": {
          "holdout": 0.06570540944836432,
          "train": 0.16862925109687887,
          "validation": 0.16027201967359933
        },
        "split": {
          "holdout": [
            "2026-02-02",
            "2026-10-23",
            183
          ],
          "train": [
            "2023-11-16",
            "2025-05-06",
            365
          ],
          "validation": [
            "2025-05-07",
            "2026-01-30",
            183
          ]
        },
        "window_r": [
          0.17877451218767237,
          0.09997034286986964,
          0.2786393540892367,
          0.15356939341852874
        ]
      }
    }
  },
  "warnings": [
    "Exploratory findings: hypotheses were generated and screened on the same data universe."
  ]
}
```

## Strategy simulation (`POST /api/research/runs/<id>/simulate`)

Implements the frozen rule as a **paper trade** with your starting capital and compares it with holding the S&P 500 (and with holding the hypothesis' own asset). Benchmark: SPY (split/dividend-adjusted) if Alpaca works, otherwise the FRED `SP500` price index (price return only, dividends excluded, stated in `benchmark.note`); in demo mode a synthetic market index. It replays the stored research series (no refit) with the stated cost per unit of turnover. `period: "holdout"` (default) is the sealed out-of-sample 25%; `development` is flattering by construction; `full` mixes both. Response: `strategy`, `asset_buy_hold`, `benchmark_buy_hold` (`final_value, total_return_pct, cagr_pct, ann_vol_pct, sharpe, max_drawdown_pct, best_day_pct, worst_day_pct`), `vs_benchmark` (`excess_return_pct`, `excess_final_value`, `active_return_ci_95` block-bootstrap, `beta`, `correlation`, `verdict`: `outperformed` only if ahead AND the CI of mean daily active return is above 0, `inconclusive` if ahead but the CI includes 0, else `underperformed`), `current_signal` (what the rule would hold after the latest observation; not advice), `curve[{date, strategy, asset, benchmark, position}]` in dollars, `caveats`. Only hypotheses that completed the full pipeline with a tradable return outcome can be simulated; a volatility outcome returns 400.

## Can Gemini look things up itself?

Gemini has a Google Search grounding tool, but (a) on the free tier used here it returns HTTP 429 `RESOURCE_EXHAUSTED` (verified: plain calls work, search-grounded calls do not), and (b) prices and macro series must come from verifiable, reproducible data adapters, never from model recall or web snippets. So Gemini chooses variables and writes explanations; the server fetches data (FRED, Tiingo/Alpaca) and papers (OpenAlex / Semantic Scholar / arXiv). Grounded search could be added as another literature source if billing is enabled.

## Stock / ETF data providers

Stock and ETF series are routed to the first working provider: **Tiingo** (your own `TIINGO_API_KEY`), then **Alpaca** (optional fallback, e.g. the keys the market-map tab already uses). Each provider's key is verified once with a cached probe; a rejected or missing key makes it unusable and the next one is tried. `/api/research/catalog` -> `providers.equity_data{usable, provider, status}` and `/api/research/health` -> `equity_data_usable` show the result. With no working provider, tickers are hidden from the AI and only FRED macro series and the S&P 500 / NASDAQ / DJIA indices can be researched. Gemini never fetches data itself: it only chooses variables from the catalogue (or valid tickers); the server fetches everything through these adapters. Keyless scraping sources (e.g. Stooq, Yahoo) are deliberately not used: Stooq now blocks scripted access and Yahoo has no licensed API. Tiingo has been tested against mocked HTTP only (no key was available in development).

## Alpaca credentials

The engine now **verifies Alpaca keys with one cached probe** (`research/providers.py: AlpacaProvider.check`). Keys that are set but rejected (401/403: revoked or rotated) are treated as unusable: the catalogue marks every ticker unavailable with the reason `Alpaca rejected the credentials`, the AI is only offered FRED variables, and requests naming tickers get a 400 with that reason, instead of hypotheses failing late with "data screening failed". `/api/research/catalog` and `/api/research/health` expose `alpaca.usable`.

## Research tab (frontend)

`frontend/src/components/ResearchLauncher.jsx` (inputs + polling) renders `components/research/Report.jsx`, an animated step-by-step story per hypothesis: (1) correlation scatter with a counting r value, a drawing fit line, rolling-r and lag bars; (2) causation vs coincidence: checks, AI reasoning and an animated causal graph; (3) validation: gauges for tests passed, holdout hit rate and out-of-sample |r|, plus the animated holdout equity curve vs buy-and-hold; (4) peer-asset bars and table; (5) literature cards and the red-team critique with a confidence meter; (6) "Put the strategy to work": choose capital, period and cost, then see the dollar equity curves of the strategy vs the S&P 500 vs the asset, a comparison table and the current signal. Each hypothesis opens with "What related research suggests" (realism score, summary, cited papers). Hypotheses with no correlation are grouped under "Initial hypotheses with no correlation" and show only their correlation graph. Plain SVG + CSS, no chart dependency; animations respect `prefers-reduced-motion`. Use mode "Demo / test" to see every panel populated (the planted effect passes its backtest and partly generalises; the null fails; the confounded pair is contradicted).

## Tests and what was / was not verified

`pytest tests -q`: 133 tests, offline (fake Gemini, stub FRED). They cover: null / planted-signal / common-factor scenarios, transform maths and no-look-ahead, mismatched-calendar alignment, monthly-series refusal, stages 1-2 blind to the holdout, holdout evaluated once and isolated, run-level Holm family, malformed / unsafe / over-budget Gemini output and labelled fallbacks, provider outages / retry / cache / Alpaca pagination, request validation and limits, API contract, legacy presets, secrets never leaked.

* **Verified live (2026-11-07):** Gemini hypothesis generation, causal proposals, self-critique and narratives; FRED public CSV; OpenAlex literature retrieval; backtests and the full pipeline over HTTP with the real Flask app. An earlier run also verified **Alpaca IEX adjusted bars** (SPY, QQQ, JETS, XLE). On the final check the Alpaca keys in `backend/.env` returned **HTTP 401** (revoked or rotated after being committed to git), so peer/ticker tests against Alpaca were not re-verified; the engine reported this as a clear screening failure.
* **Peer generalisation and the critique's use of papers** were verified on synthetic data (peer replication) and live (critique over retrieved papers). With only FRED series, live peers exist for the equity indices (S&P 500 / NASDAQ / DJIA); no live run in this session had a detected association, so no live peer table was produced.
* **NOT verified live:** the keyed FRED API path (`FRED_API_KEY`).

## Limitations

* Observational daily data, linear methods only; `evidence_consistent` is never proof. Unobserved confounders cannot be excluded.
* Hypotheses are generated and tested on the same data universe, so results are exploratory; the holdout is only 25% of 1-5 years of daily data (low power).
* FRED values are latest-revised (no vintages), valid for market-based daily series only.
* Single-process job pool, no auth or per-user rate limiting beyond 3 active runs. Gemini output can vary between runs.
* `GET /api/health` is the host app's own endpoint; use `/api/research/health` for provider status.
