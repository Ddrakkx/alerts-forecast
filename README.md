# alerts-forecast

Probabilistic forecast of **air-raid alert onsets in Poltava oblast** (Ukraine), built as a mini-project for the
KSE Agentic AI School selection. The point of the project is an honest, leak-free evaluation, not a big number.

**Коротко українською.** Прогнозуємо ймовірність того, що в Полтавській області *почнеться* тривога в найближчі
1, 3 або 6 годин. Валідація лише walk-forward, оцінка на останніх 8 тижнях, похибки блочним бутстрепом по днях.
Надійний результат один: при горизонті 1 година моделі (логістична регресія, бустинг) **краще впорядковують моменти**,
ніж бейзлайни (PR-AUC на 0,05–0,08 вище), і це відтворюється на другому джерелі даних. **Якість ймовірностей
(Brier) значущо не покращилась ніде**, а при 3 та 6 годинах переваги над найкращим бейзлайном немає або вона не доведена.
Журнал усіх рішень, помилок і відкликаних висновків: [docs/decisions.md](docs/decisions.md).

## Main result

Test block = last 8 weeks (2026-08-14 .. 2026-10-09), 57 Kyiv days, about 4,500 forecast moments per horizon.
Mean and 95% interval from a bootstrap that resamples whole days. The configuration of every row was chosen on the
8 weeks before the test block (see "Validation protocol").

| Horizon | Model | Brier ↓ | PR-AUC ↑ |
|---|---|---|---|
| **H = 1 h** (30.0% positive) | constant rate (best baseline) | 0.2106 [0.1977, 0.2239] | 0.271 [0.240, 0.313] |
| | recent activity | 0.2108 [0.1978, 0.2243] | 0.285 [0.252, 0.326] |
| | hour of week | 0.2190 [0.2041, 0.2347] | 0.289 [0.257, 0.329] |
| | logistic, own history | 0.2098 [0.1977, 0.2228] | 0.323 [0.280, 0.372] |
| | logistic, + neighbours + country | 0.2095 [0.1980, 0.2216] | 0.336 [0.297, 0.380] |
| | boosting, + neighbours + country | 0.2085 [0.1968, 0.2207] | 0.351 [0.302, 0.405] |
| **H = 3 h** (66.3% positive, main) | smoothed hour of week (post hoc, best baseline) | 0.2241 [0.2099, 0.2385] | 0.659 [0.612, 0.711] |
| | constant rate | 0.2243 [0.2100, 0.2387] | 0.658 [0.599, 0.722] |
| | hour of week | 0.2377 [0.2243, 0.2543] | 0.670 [0.618, 0.721] |
| | logistic, + neighbours + country | 0.2237 [0.2067, 0.2404] | 0.692 [0.639, 0.740] |
| | boosting, + neighbours + country | 0.2260 [0.2090, 0.2424] | 0.676 [0.620, 0.736] |

Paired differences to the best baseline (same resampled days; PR-AUC positive and Brier negative mean better):

| | PR-AUC | Brier |
|---|---|---|
| H=1, logistic regression, all features | **+0.066 [0.034, 0.095]** | -0.0012 [-0.0038, 0.0012] |
| H=1, boosting, all features | **+0.080 [0.040, 0.119]** | -0.0022 [-0.0053, 0.0007] |
| H=3, logistic regression, all features | +0.033 [-0.008, 0.069] | -0.0004 [-0.0046, 0.0040] |
| H=3, boosting, all features | +0.017 [-0.028, 0.063] | +0.0018 [-0.0029, 0.0063] |

What this says, and what it does not:

- **H = 1: the models rank moments better than any baseline.** The gain holds with moments that depend on an invented alert end
  removed, without the last 24 h of data, and on an earlier period for both the volunteer and the official source
  (`results/robustness.txt`).
- **Probabilities did not get better.** No Brier difference to the best baseline has an interval that excludes zero, at any horizon.
  Platt recalibration did not help either.
- **H = 3 and H = 6: no demonstrated gain** over the best baseline. H = 6 is almost uninformative (89% of moments are positive,
  a constant already has PR-AUC 0.889).
- **Boosting vs logistic regression:** on par. Against the logistic variant chosen on validation it looked better at H=1
  (+0.052 [0.022, 0.085] PR-AUC), but against the best-performing logistic variant the difference is +0.015 [-0.013, 0.046].
- **The hour of the week carries no usable signal** beyond the base rate: validation picks very strong smoothing for it
  (practically the constant), and the raw version has a *worse* Brier than the constant at both horizons.
- Neighbouring oblasts and country-wide activity helped boosting at H=1 (post hoc, exploratory: +0.051 [0.017, 0.088] PR-AUC over
  the same model on own history), not logistic regression.

## Task definition

- **Moment `t`**: every 15 minutes on the UTC clock. Only alerts with `started_at <= t` are known at `t`.
- **Target** `y_H(t) = 1` if a new alert **starts** in `(t, t + H]`, for `H` in {1, 3, 6} hours (main: 3). A start exactly at `t` is the past.
- **Main sample**: only moments with **no alert active** at `t` (an alert is active if `started_at <= t < finished_at`; only the fact
  "not finished yet" is used, never the value of `finished_at`). On the other moments the plain "same as now" baseline would be trivial;
  instead the persistence baseline is "an alert ended within the last `H` hours".
- **End of data** = the latest timestamp anywhere in the file (2026-10-09 05:12:41 UTC); the last usable `t` is `end - H`.
- Positive rate is not stable: for H=3 it ranges 17-59% by month in 2022-2025 and 61-68% since May 2026 (`scripts/target_by_month.py`). This is a real change
  (alerts per month about doubled between spring 2025 and spring 2026), so training windows are rolling and chosen on validation.

## Data

[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT), snapshot
`2f115548a8bd0816c901dbf422082091b3e5c34c` of 2026-10-09, pinned in `scripts/download_data.py`. Times are UTC (checked in the files).

- **Primary: the volunteer file** (eTryvoga channel, unofficial, oblast level for the whole period). 1.9% of Poltava alerts have an
  *invented* end (`naive`, start + 30 min); they are kept and a sensitivity run removes the moments they touch.
- **Check: the official file.** Since late 2025 it holds raion-level alerts only, ends 2026-09-07, has a hole at 2026-08-29..09-02,
  and every oblast-level row is duplicated. It is deduplicated and raions are merged into oblast episodes. The two sources agree in 2023
  (885 vs 886 alerts) and differ in 2026 (the official merge gives ~14% fewer alerts because overlapping raion alerts form one long episode),
  so only the direction of results is compared, not levels.
- Luhansk oblast is excluded from country-wide counts (permanent siren, not a series). Crimea is not in the data.
- Raw data is not stored in the repository: `python scripts/download_data.py`.

## Leakage controls

- **Truncation test** (`src/alerts_forecast/leakage.py`, `tests/test_leakage.py`, `tests/test_features.py`): for random moments and for the
  nasty ones (exactly at starts and ends of alerts), cut the data at `t` (drop later alerts, hide ends after `t`), recompute the features and
  require identical values. It runs on synthetic and on real data, over all regions at once, and it is shown to **catch** a deliberately leaky
  feature ("minutes until the alert ends").
- **No "time left" features** anywhere; time under alert uses only the part before `t`.
- **Walk-forward with a purge gap**: training rows need `t + H <= start of the test block`, otherwise their labels would look into the test block.
- Scalers and all fitted parameters use training rows only; tests check that predictions do not change when test labels are flipped.
- Naive-end flags are used only to filter the evaluation, never as a feature.
- Local time (Kyiv, with summer time) is used for calendar features only; everything else is UTC.

## Validation protocol

- Test block: last 8 weeks. Validation block: the 8 weeks before. Weekly walk-forward blocks, refit every week.
- For every model family a small grid is searched (training window all/365/180/90 days; `k` of the smoothing; `C` of the regression;
  depth and iterations of boosting). The best configuration of each family is chosen by **Brier on the validation block**; the test block is
  then reported for the chosen ones.
- Uncertainty: bootstrap over whole Kyiv days (1000 draws, fixed seed), also for paired differences.
- Metrics: PR-AUC, Brier score, reliability table and calibration slope/intercept.
- Baselines required by the task: "as now" (persistence, adapted as above) and "mean by hour of week". Extra: constant rate, and a smoothed hour of week
  (**added post hoc**, after the first test results were seen; the bar for the models is the best baseline on validation).

## Features (`src/alerts_forecast/features.py`)

- Own history: time since the last alert ended, starts in the last 3 h / 24 h / 7 d, share of the last 24 h under alert, Kyiv time of day, weekend.
- Seven neighbouring oblasts (Chernihiv, Sumy, Kharkiv, Dnipropetrovsk, Kirovohrad, Cherkasy, Kyiv oblast): active flags, starts in the last 1 h / 3 h,
  time since their latest start. The list comes from my knowledge of the map, not from the data.
- Country-wide: number of oblasts under alert, starts in the last 1 h / 3 h / 24 h.

## Limitations (please read)

- **The test block is not untouched.** It was looked at for the baselines and logistic regression several times, and the choices made after seeing it are
  listed in `docs/decisions.md` (smoothed baseline, wider grids, post-hoc comparisons). Boosting was looked at once with a success criterion written down
  before the run. A later data snapshot would give a block nobody has seen.
- Neighbouring rows (15 minutes apart) are near duplicates, so the real number of independent observations is far smaller than the row count; the intervals are wide for that reason.
- The best `C` of the regression is the smallest in the grid: validation keeps asking for predictions close to the constant.
- The volunteer file is unofficial and splits alerts differently from the official one. It may also lag near its end (cannot be checked; a run without the last 24 h gives the same conclusions).
- Many comparisons were printed; only "H=3, best model vs best baseline" was planned. The rest is exploratory.
- First results with a narrow grid (`results/experiment_v1_narrow_grid_superseded.txt`) showed a significant H=3 gain that **disappeared** with the wider grid; it is withdrawn.

## How to reproduce

Python 3.12, tested on Windows 11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # about 40 MB, pinned snapshot, sizes are checked
python -m pytest                  # about 30 s
python scripts/run_experiment.py  # baselines + logistic regression, H = 3, 1, 6   (a few minutes)
python scripts/run_robustness.py  # naive / last day / official source
python scripts/run_boosting.py    # gradient boosting, H = 1, 3
```

Results are deterministic (fixed seeds). The outputs of the runs used in this README are in `results/`.

## Layout

| Path | Content |
|---|---|
| `src/alerts_forecast/data.py` | loading, validation, official file as oblast episodes |
| `src/alerts_forecast/target.py` | 15-minute grid, state at `t`, onset label, main sample |
| `src/alerts_forecast/features.py` | own, neighbour and country features |
| `src/alerts_forecast/baselines.py`, `models.py` | baselines, logistic regression, Platt scaling, boosting |
| `src/alerts_forecast/walkforward.py`, `experiment.py` | folds with purge gap, grids, validation-based choice |
| `src/alerts_forecast/metrics.py`, `leakage.py` | PR-AUC, Brier, calibration, day-block bootstrap; truncation check |
| `scripts/` | download, monthly target rates, the experiment runners; `run_baselines.py` is the first baseline-only run (decision 3) |
| `tests/` | 46 tests, including the leakage checks |
| `results/` | outputs of the runs |
| `docs/decisions.md` | decision log: options, choices, risks, mistakes and retractions |

## Process and AI use

The project was built with Claude Code (Claude Sonnet 5.5) as the main engineering tool, step by step: plan, data checks, target and grid, leakage tests,
baselines, features, models, robustness. The decision log records, for each step, what was proposed, what was checked in the data, what was wrong and how it was
corrected (for example: the first end-of-data choice, the first estimate of the hole in the official file, a bug on empty histories found by the truncation
test, tests that passed by accident, and the withdrawn H=3 result). The full dialogue is submitted separately.

## License

Code: MIT (see `LICENSE`). The data keeps the license of its source repository (MIT) and is not redistributed here.

## Possible extensions

- More oblasts (Kyiv, others) with the same pipeline; the region is a single constant today.
- A later data snapshot as a really untouched test block.
- News, official statements and public Telegram channels, only items published before the forecast moment.
- Choosing configurations by PR-AUC or by a combination instead of Brier alone.
