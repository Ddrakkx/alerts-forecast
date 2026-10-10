# alerts-forecast

Probabilistic forecast of **air-raid alert onsets in Ukrainian oblasts** (main: Poltava; for comparison: Kyiv oblast, Kharkiv
oblast, Lviv oblast), built as a mini-project for the KSE Agentic AI School selection. The point of the project is an honest,
leak-free evaluation, not a big number.

**Коротко українською.** Прогнозуємо ймовірність того, що в області *почнеться* тривога в найближчі 1, 3 або 6 годин.
Валідація лише walk-forward, оцінка на останніх 8 тижнях, похибки блочним бутстрепом по днях. Кожна область оцінюється окремо.
Головний результат чесно негативний для основної області: **у Полтавській моделі (логістична регресія, бустинг) не обігрують
найсильніший простий бейзлайн** («частота тривог за останні 30 днів, оновлювана щодня») ні за PR-AUC, ні за Brier.
У Київській, Харківській і Львівській моделі його обігрують за PR-AUC, а подекуди й за Brier (найчіткіше при горизонті 1 год).
Під час роботи виявилось, що спершу бейзлайни були надто слабкі (статичні); це виправлено і докладно описано.
Журнал усіх рішень, помилок і відкликаних висновків: [docs/decisions.md](docs/decisions.md).

## Main results

Test block = last 8 weeks (2026-08-14 .. 2026-10-09, 57 Kyiv days, about 4,500 forecast moments per oblast and horizon).
Intervals are 95% from a bootstrap that resamples whole days. Differences are **paired** (same resampled days): PR-AUC positive and
Brier negative mean the model is better. The configuration of every model was chosen on the 8 weeks before the test block.

The reference below is the **best baseline on the test block itself** ("strict": chosen with hindsight, so it favours the baselines).
A second table in [results/summary.md](results/summary.md) uses the baseline chosen on the validation block instead.
† marks an interval that excludes zero (judged on the rounded numbers).

| Oblast | H | positive in test / validation | strict reference | its Brier / PR-AUC | logistic regression: dPR-AUC | dBrier | boosting: dPR-AUC | dBrier |
|---|---|---|---|---|---|---|---|---|
| **Poltavska** (main) | 1 | 30.0% / 31.0% | recent level | 0.2100 / 0.315 | +0.021 [-0.016, +0.050] | -0.0005 [-0.0026, +0.0016] | +0.035 [-0.006, +0.071] | -0.0016 [-0.0044, +0.0013] |
| | 3 | 66.3% / 66.9% | recent level | 0.2233 / 0.686 | +0.006 [-0.041, +0.053] | +0.0004 [-0.0029, +0.0039] | -0.009 [-0.059, +0.046] | +0.0027 [-0.0017, +0.0069] |
| Kyivska | 1 | 40.1% / 16.2% | recent level | 0.2351 / 0.470 | +0.102 [+0.060, +0.146] † | -0.0168 [-0.0270, -0.0082] † | +0.088 [+0.051, +0.124] † | -0.0083 [-0.0145, -0.0020] † |
| | 3 | 73.9% / 39.0% | recent level | 0.1802 / 0.808 | +0.056 [+0.019, +0.082] † | -0.0047 [-0.0169, +0.0075] | +0.059 [+0.028, +0.083] † | +0.0041 [-0.0062, +0.0142] |
| Kharkivska | 1 | 63.1% / 57.2% | smoothed hour of week | 0.2309 / 0.702 | +0.034 [+0.003, +0.059] † | -0.0092 [-0.0139, -0.0043] † | +0.045 [+0.016, +0.074] † | -0.0097 [-0.0146, -0.0047] † |
| | 3 | 93.0% / 90.7% | smoothed hour of week | 0.0644 / 0.945 | +0.016 [+0.001, +0.035] † | -0.0009 [-0.0019, -0.0000] | +0.006 [-0.012, +0.023] | -0.0004 [-0.0016, +0.0007] |
| Lvivska | 1 | 17.5% / 12.8% | recent level | 0.1446 / 0.201 | +0.041 [+0.004, +0.074] † | -0.0027 [-0.0060, +0.0003] | +0.031 [+0.001, +0.065] † | -0.0018 [-0.0049, +0.0011] |
| | 3 | 41.9% / 33.2% | recent level | 0.2466 / 0.453 | +0.071 [+0.012, +0.116] † | -0.0116 [-0.0244, -0.0006] † | +0.045 [-0.009, +0.092] | -0.0053 [-0.0175, +0.0064] |

H = 6 is in `results/summary.md`; it is nearly uninformative (88-99% of moments are positive in three of the four oblasts).

What this says, and what it does not:

- **Poltava (main oblast): no demonstrated gain** over the strongest simple baseline, at any horizon, for either model. All eight intervals
  contain zero. The raw numbers (below) are close to a coin flip between the models and the "recent level" baseline.
- **Kyiv, Kharkiv, Lviv: the models rank moments better** than even the strict reference at H=1 (PR-AUC +0.03 to +0.10) and, in several cases,
  also give better probabilities (Brier: Kyiv H=1, Kharkiv H=1, Lviv H=3). At H=3 the gain is smaller or uncertain.
- **Probabilities (Brier) are the weaker half** of the story: out of 16 model-horizon cells above, 5 have a Brier interval excluding zero.
- **Boosting is on par with logistic regression** (no cell where it is demonstrably better).
- **The hour of the week carries little usable signal in Poltava** (the smoothed version collapses to the base rate), but real signal in Kharkiv: there the smoothed hour of week is the strongest baseline.
- Neighbouring oblasts and country-wide activity helped boosting at H=1 in Poltava (post hoc, exploratory), not logistic regression; see `docs/decisions.md`.

### Poltava in detail (test block, mean and 95% interval)

| Horizon | Model | Brier ↓ | PR-AUC ↑ |
|---|---|---|---|
| **H = 1 h** (30.0% positive) | constant rate (static, 90 d) | 0.2106 [0.1977, 0.2239] | 0.271 [0.240, 0.313] |
| | recent activity ("alert ended in the last H hours") | 0.2108 [0.1978, 0.2243] | 0.285 [0.252, 0.326] |
| | hour of week | 0.2190 [0.2041, 0.2347] | 0.289 [0.257, 0.329] |
| | **recent level (last 30 days, refit daily; post hoc)** | 0.2100 [0.1978, 0.2229] | 0.315 [0.273, 0.373] |
| | logistic, own history | 0.2098 [0.1977, 0.2228] | 0.323 [0.280, 0.372] |
| | logistic, + neighbours + country | 0.2095 [0.1980, 0.2216] | 0.336 [0.297, 0.380] |
| | boosting, + neighbours + country | 0.2085 [0.1968, 0.2207] | 0.351 [0.302, 0.405] |
| **H = 3 h** (66.3% positive) | constant rate | 0.2243 [0.2100, 0.2387] | 0.658 [0.599, 0.722] |
| | hour of week | 0.2377 [0.2243, 0.2543] | 0.670 [0.618, 0.721] |
| | **recent level (last 30 days, refit daily; post hoc)** | 0.2233 [0.2074, 0.2390] | 0.686 [0.616, 0.752] |
| | logistic, + neighbours + country | 0.2237 [0.2067, 0.2404] | 0.692 [0.639, 0.740] |
| | boosting, + neighbours + country | 0.2260 [0.2090, 0.2424] | 0.676 [0.620, 0.736] |

The point estimates of the models are higher in PR-AUC at H=1, but the paired differences to the recent-level baseline have intervals that include zero.

## How the conclusion changed during the work (read this)

An earlier version of this README said that at H=1 the models clearly beat the baselines in Poltava (PR-AUC +0.066 [0.034, 0.095]). That was true
against the baselines I had at the time and it is **withdrawn**:

1. The first baselines had static training windows (at least 90 days, refit weekly). In Poltava the alert rate is stable, so this looked fine.
2. When Kyiv, Kharkiv and Lviv were added, Kyiv showed huge Brier gains. The positive rate there jumps between validation and test (H=1: 16% to 40%).
   A diagnostic showed that a plain "rate of the last 14 days, refit daily" gets most of that gain without any model.
3. That baseline (**post hoc**, `recent_level`) was added for every oblast, Poltava included. Against it the Poltava advantage disappears.
4. Because the baseline chosen on validation can fail under a regime change (in Kyiv validation chose a static one that broke on the test), the strict
   reference "best baseline on the test block" was added as well.

The same pattern happened earlier with a narrow hyperparameter grid (a significant H=3 gain disappeared with a wider grid) and is logged in `docs/decisions.md`.

## Task definition

- **Moment `t`**: every 15 minutes on the UTC clock. Only alerts with `started_at <= t` are known at `t`.
- **Target** `y_H(t) = 1` if a new alert **starts** in `(t, t + H]`, for `H` in {1, 3, 6} hours (main: 3). A start exactly at `t` is the past.
- **Main sample**: only moments with **no alert active** at `t` (an alert is active if `started_at <= t < finished_at`; only the fact
  "not finished yet" is used, never the value of `finished_at`). On the other moments the plain "same as now" baseline would be trivial;
  instead the persistence baseline is "an alert ended within the last `H` hours".
- **End of data** = the latest timestamp anywhere in the file (2026-10-09 05:12:41 UTC); the last usable `t` is `end - H`.
- The positive rate is not stable: for Poltava H=3 it ranges 17-59% by month in 2022-2025 and 61-68% since May 2026 (`scripts/target_by_month.py --region ...`).

## Data

[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT), snapshot
`2f115548a8bd0816c901dbf422082091b3e5c34c` of 2026-10-09, pinned in `scripts/download_data.py`. Times are UTC (checked in the files).

- **Primary: the volunteer file** (eTryvoga channel, unofficial, oblast level for the whole period). 1.9% of Poltava alerts have an
  *invented* end (`naive`, start + 30 min); they are kept and a sensitivity run removes the moments they touch.
- **Check: the official file.** Since late 2025 it holds raion-level alerts only, ends 2026-09-07, has a hole at 2026-08-29..09-02,
  and every oblast-level row is duplicated. It is deduplicated and raions are merged into oblast episodes. For Poltava and Kyiv the volunteer and
  official sources give results in the same direction (`results/*/robustness.txt`). For **Kharkiv and Lviv the official source is not usable
  as a check**: after merging raions Kharkiv oblast is almost always "under alert" (386 moments without alert in the check period, against
  about 4,200 in the volunteer data) and Lviv has almost no official alerts in that period (0.9% positive against 38% in the volunteer data).
  The sources differ in how an alert is split, so levels are never compared, only the direction.
- Luhansk oblast is excluded from country-wide counts (permanent siren, not a series). Crimea is not in the data.
- **Why Lviv and not Zakarpattia**: Zakarpatska oblast has 0 alerts in the last 8 weeks (Ivano-Frankivsk 8, Ternopil 21), so there is nothing to test; Lviv has 245
  (Poltava 382) and is the quietest western oblast that can be evaluated.
- Raw data is not stored in the repository: `python scripts/download_data.py`.

## Leakage controls

- **Truncation test** (`src/alerts_forecast/leakage.py`, `tests/test_leakage.py`, `tests/test_features.py`): for random moments and for the
  nasty ones (exactly at starts and ends of alerts), cut the data at `t` (drop later alerts, hide ends after `t`), recompute the features and
  require identical values. It runs on synthetic and on real data, over all regions at once, for more than one oblast, and it is shown to **catch**
  a deliberately leaky feature ("minutes until the alert ends").
- **No "time left" features** anywhere; time under alert uses only the part before `t`.
- **Walk-forward with a purge gap**: training rows need `t + H <= start of the test block`, otherwise their labels would look into the test block.
- Scalers and all fitted parameters use training rows only; tests check that predictions do not change when test labels are flipped.
- Naive-end flags are used only to filter the evaluation, never as a feature.
- Local time (Kyiv, with summer time) is used for calendar features only; everything else is UTC.

## Validation protocol

- Test block: last 8 weeks. Validation block: the 8 weeks before. Weekly walk-forward blocks, refit every week (the `recent_level` baseline: every day).
- Each oblast is run completely **separately**: its own neighbour list, its own hyperparameters chosen on its own validation block, its own references.
- For every model family a small grid is searched (training window all/365/180/90 days; `k` of the smoothing; `C` of the regression; depth and iterations of boosting;
  window of the recent level 3/7/14/30 days). The best configuration of each family is chosen by **Brier on the validation block**; the test block is then reported
  for the chosen ones.
- References: "as now" (persistence, adapted as above), "mean by hour of week" (both required by the task), a constant rate, and two **post-hoc** baselines
  (smoothed hour of week; recent level). Two bars are reported: the best baseline on validation, and the best baseline on the test block (strict).
- Uncertainty: bootstrap over whole Kyiv days (1000 draws, fixed seed), also for paired differences.
- Metrics: PR-AUC, Brier score, reliability table and calibration slope/intercept (slopes of step-wise constant baselines are meaningless).

## Features (`src/alerts_forecast/features.py`)

- Own history: time since the last alert ended, starts in the last 3 h / 24 h / 7 d, share of the last 24 h under alert, Kyiv time of day, weekend.
- Neighbouring oblasts: active flags, starts in the last 1 h / 3 h, time since their latest start. The lists come from my knowledge of the map, not from the data
  (`NEIGHBORS_BY_REGION`): Poltava (Chernihiv, Sumy, Kharkiv, Dnipropetrovsk, Kirovohrad, Cherkasy, Kyiv oblast), Kyiv oblast (Zhytomyr, Chernihiv, Poltava, Cherkasy,
  Vinnytsia, Kyiv City), Kharkiv (Sumy, Poltava, Dnipropetrovsk, Donetsk), Lviv (Volyn, Rivne, Ternopil, Ivano-Frankivsk, Zakarpattia).
- Country-wide: number of oblasts under alert, starts in the last 1 h / 3 h / 24 h.

## Limitations (please read)

- **The test block is not untouched.** It was looked at several times, and three baselines were added or changed after seeing results: the smoothed hour of week, the
  recent level, and the strict reference. Boosting was looked at once with a success criterion written before the run. A later data snapshot would give a block nobody has seen.
- **Choosing a configuration on the validation block is fragile when the alert rate changes** (Kyiv). That is why the strict reference exists; it is optimistic for the baselines.
- The daily-refit baseline is updated more often than the weekly-refit models. That makes it a stronger bar, which is the conservative direction for the claims.
- Neighbouring rows (15 minutes apart) are near duplicates, so the number of independent observations is far smaller than the row count; intervals are wide for that reason.
- For the logistic regression with all features the best `C` is 0.001 (the smallest of the grid) in 10 of 12 oblast-horizon cells and 0.003 in the other two: validation keeps asking for predictions close to a simple rate.
- The volunteer file is unofficial and splits alerts differently from the official one. It may also lag near its end (cannot be checked; a run without the last 24 h gives the same conclusions).
- `robustness.txt` (no naive-affected moments, no last 24 h, official source) compares with the baseline chosen on validation, not with the strict one.
- Many comparisons were printed; only "Poltava H=3, best model vs best baseline" was planned. The rest, including all new oblasts, is exploratory.
- Neighbour lists are from memory of the map.

## How to reproduce

Python 3.12, tested on Windows 11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # about 40 MB, pinned snapshot, sizes are checked
python -m pytest                  # about 1 minute, 50 tests
bash scripts/run_region.sh "Poltavska oblast"   # experiment + robustness + boosting for one oblast, about 10 minutes
bash scripts/run_region.sh "Kyivska oblast"     # also: Kharkivska oblast, Lvivska oblast
python scripts/summarize_regions.py             # builds results/summary.md from the saved outputs
```

Results are deterministic (fixed seeds). The outputs of the runs used in this README are in `results/<oblast>/`.

## Layout

| Path | Content |
|---|---|
| `src/alerts_forecast/data.py` | loading, validation, official file as oblast episodes |
| `src/alerts_forecast/target.py` | 15-minute grid, state at `t`, onset label, main sample |
| `src/alerts_forecast/features.py` | own, neighbour and country features; neighbour lists |
| `src/alerts_forecast/baselines.py`, `models.py` | baselines, logistic regression, Platt scaling, boosting |
| `src/alerts_forecast/walkforward.py`, `experiment.py` | folds with purge gap, grids, validation-based choice, strict reference |
| `src/alerts_forecast/metrics.py`, `leakage.py` | PR-AUC, Brier, calibration, day-block bootstrap; truncation check |
| `scripts/` | download, per-oblast runner, experiment/robustness/boosting runners, summary; `run_baselines.py` is the first baseline-only run (decision 3) |
| `tests/` | 50 tests, including the leakage checks |
| `results/<oblast>/`, `results/summary.md` | outputs of the runs; `poltavska/experiment_v1_narrow_grid_superseded.txt` is an old, withdrawn run |
| `docs/decisions.md` | decision log: options, choices, risks, mistakes and retractions |

## License

Code: MIT (see `LICENSE`). The data keeps the license of its source repository (MIT) and is not redistributed here.

## Process and AI use

The project was built with Claude Code (Claude Sonnet 5.5) as the main engineering tool, step by step: plan, data checks, target and grid, leakage tests,
baselines, features, models, robustness, more oblasts. The decision log records, for each step, what was proposed, what was checked in the data, what was
wrong and how it was corrected (for example: the first end-of-data choice, the first estimate of the hole in the official file, a bug on empty histories found by
the truncation test, tests that passed by accident, the withdrawn Poltava results, and the baselines that turned out to be too weak). The full dialogue is submitted separately.

## Possible extensions

- More oblasts with the same pipeline (add the neighbour list to `NEIGHBORS_BY_REGION`, then `scripts/run_region.sh`).
- A later data snapshot as a really untouched test block.
- Models that are refit more often than weekly, or use the recent level as a feature.
- News, official statements and public Telegram channels, only items published before the forecast moment.
- Choosing configurations by PR-AUC or by a combination instead of Brier alone.
