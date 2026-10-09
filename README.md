# alerts-forecast

Probabilistic forecast of air-raid alert onsets in Poltava oblast (Ukraine),
built as a mini-project for the KSE Agentic AI School selection.

## Task

At time `t`, predict the probability that a new alert **starts** in the window
`(t, t+H]` for `H` in {1h, 3h, 6h}. The main evaluation uses only moments `t`
when no alert is active. Nothing after `t` may be used for training features
or inputs (no data leakage).

## Data

[ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(MIT license, times in UTC). Raw files are not stored here; see `data/README.md`.

## Method (planned)

- Baselines: "recent activity" (alert in the last H hours) and "mean by hour of week".
- Validation: walk-forward with a purge gap of at least `H`.
- Metrics: PR-AUC, Brier score, calibration.

## How to run

TODO

## Possible extensions

- News, official statements and public Telegram channels as extra features
  (only items published before the forecast moment).
- Neighbouring oblasts and gradient boosting.
- Kyiv as a second region.

## Process

Decisions, mistakes and fixes are logged in `docs/decisions.md`.
