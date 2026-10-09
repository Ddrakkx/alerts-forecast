# Decision log

Template for each entry:

## <number>. <decision>
- Options:
- Choice and why:
- Risks:
- Date / what changed later:

## 1. Data source: volunteer as primary, official as robustness check
- Options: official dataset, volunteer dataset (eTryvoga), or both.
- Facts checked on snapshot 2f11554 (Poltavska oblast):
  - Official ends 2026-09-07 (volunteer: 2026-10-08), has a hole of 3 empty days
    (2026-08-30..09-01, whole country), has no oblast-level rows after 2025 (raion only)
    and every oblast-level row is duplicated.
  - Volunteer is oblast-level for the whole period, naive (invented end, +30 min) share is 1.9%.
  - 2023: after dedup the sources match (885 vs 886 alerts, 99% of starts within 5 min).
  - 2026-03..09: official raion rows merged into oblast episodes give ~14% fewer
    alerts than volunteer. Not caused by merge gap (ratio grows with gap: 1.14 -> 1.34)
    or by the last day (excluded). Almost all extra volunteer starts (Mar-Jul: 100%)
    fall inside an official merged episode, i.e. a definition difference: overlapping
    raion alerts form one long official episode, volunteers split them.
    45 volunteer starts have no official alert at all; checked by date: 41 of them fall on
    2026-08-29..09-02 (a wider official hole in Poltava than the 3 fully empty days,
    which I first assumed), the other 4 are isolated (07-04, 07-27, 08-08, 08-20).
- Correction: my first claim "hole = 08-30..09-01" was too narrow, found by checking dates.
- Choice: volunteer as the main source (covers the latest period, one consistent level);
  official only as a check: dedup, merge raions into oblast episodes, drop 2026-08-29..09-02.
  Compare direction and ranking of models, not absolute numbers.
- Risks: volunteer data is unofficial; its rule for splitting alerts may differ from
  the official one; this must be stated in the README.

## 2. Target, grid, sample
- Grid: forecast moments t every 15 min on the UTC clock (:00 :15 :30 :45).
  At t only alerts with started_at <= t are known. Alert is active if started_at <= t < finished_at.
  Only the fact "not finished yet" is used, never the value of finished_at.
- Target: y_H(t) = 1 if an alert starts in the half-open window (t, t+H]. A start exactly at t is the past.
  H in {1, 3, 6} hours, all reported, the main one is H = 3.
- Main sample (train and test): only t with no active alert. Active t are kept for a secondary check.
- Naive alerts (invented end = start + 30 min, 1.9%) stay as they are. Sensitivity run excludes t whose
  previous alert is naive (`prev_naive`). It is for evaluation only, never a model feature,
  because in real time it is not known that an end was invented.
- Mandatory breakdown of results by minutes since the last alert ended: <=30, 30-180, >180.
  Reason: 12% of alert starts come <=30 min after the previous end, 25% <=60 min (last year, volunteer, Poltava).
- End of data = the latest timestamp observed anywhere in the file (any oblast, started_at or finished_at),
  2026-10-09 05:12:41 UTC. Last usable t = floor_15min(end - H), so the label window is fully observed.
- Correction: I first proposed the commit time of the dataset (05:17) as the end. It says when the file
  was built, not what was observed, so it was replaced by the last observed timestamp (user's question).
- Risks: the file may lag behind real events near its end (cannot be checked); I will test dropping the last day later.

## 3. Validation scheme and first baseline results
- Final test block: the last 8 weeks, 2026-08-14 .. data end (57 Kyiv days). Validation block: the 8 weeks before it.
  Walk-forward in weekly blocks, refit every block. Train rows need t + H <= block start (purge = H, exactly what
  the label window needs; I first said 6 h for all horizons, H is the real requirement).
- Training window (all past, 365, 180, 90 days) is chosen on the validation block only, by Brier of hour_of_week.
  Choosing it on the test block would be tuning on the test.
- Uncertainty: bootstrap that resamples whole Kyiv days (1000 draws), plus paired differences on the same resampled days.
- Why rolling windows matter: positive rate of H=3 moves from ~20-35% (2022-23, 2025 spring) to 62-68% (2026-05..10).
  Not a bug: alerts per month grew from ~100-125 (2025) to 200+ (2026). `scripts/target_by_month.py`.
- Mistake in my own test: I first asserted that fold predictions do not depend on test labels by flipping the labels
  of the whole test range, but fold 2 legitimately trains on fold 1 labels (already past). Correct statement: a fold's
  predictions depend only on labels with t <= block start - H. Test rewritten (and a counter-test added).
- First results (H=3, test, 90 d window): constant Brier 0.2243, recent_activity 0.2250, hour_of_week 0.2377.
  hour_of_week is WORSE than the constant rate (paired Brier difference -0.013, interval -0.024..-0.003).
  PR-AUC of the three is statistically indistinguishable (intervals overlap, paired intervals contain 0).
  Calibration of hour_of_week: predictions span 0.46..0.84, observed 0.63..0.69 (almost flat): noisy slot means.
  H=6: positive rate 89%, a constant already gives PR-AUC 0.889, so H=6 carries almost no information.
- Caveat: I saw these test numbers before deciding anything about the baselines. Any change to a baseline
  (for example smoothing hour_of_week) has to be tuned on the validation block and reported as a post-hoc change.

## 4. Features beyond one oblast, logistic regression, first model results
- Added features (src/alerts_forecast/features.py): own history (time since last end, starts in 3h/24h/7d, share of last 24h
  under alert, Kyiv time of day, weekend), 7 neighbours (active flags, starts in 1h/3h, time since their latest start),
  country-wide (number of oblasts under alert, starts in 1h/3h/24h; Luhanska excluded: permanent siren).
  Neighbour list is from my memory of the map, not from the data: Chernihivska, Sumska, Kharkivska, Dnipropetrovska,
  Kirovohradska, Cherkaska, Kyivska. To be checked on a map.
- Leakage: the same truncation test as before, now over all regions at once. It passes on synthetic and real data and it
  catches a deliberately leaky "neighbour ends soon" feature. It also found an empty-table bug (a region with no known alert yet).
- Mistakes in my tests found on the way: a DST date (2026-03-30 is after the 03-29 switch) made a slot test pass by accident,
  because the expected value equalled the fallback value. Rewritten so the slot mean differs from the global mean.
- Model: standardised logistic regression, grid C in {0.01, 0.1, 1} x window {365, 180, 90 d}, chosen on validation by Brier.
  Post-hoc smoothed hour_of_week: k in {25, 100, 400, 1600}.
- Results (test, last 8 weeks; scripts/run_experiment.py, full output in results/experiment_2026-10-09.txt):
  - Best baseline (bar): constant rate. The smoothed hour_of_week chose k=1600, the edge of the grid, and equals the constant.
    So the hour of the week carries no usable signal in these data beyond the base rate (also at H=1, H=6).
  - H=3: PR-AUC of logreg[own+nbr+cty] 0.727 vs 0.659 for the bar, paired difference +0.069 [0.018, 0.116]. Brier 0.2218 vs 0.2241,
    paired difference -0.0024 [-0.0096, 0.0040], not distinguishable from zero. On the validation block the logistic regression was
    NOT better than the constant in Brier (0.2287 vs 0.2256). Verdict: it ranks moments better, it does not give better probabilities.
  - H=1: PR-AUC 0.352 vs 0.271 for the constant, +0.081 [0.037, 0.123]; Brier difference -0.0020 [-0.0069, 0.0023].
  - Neighbours alone are not a significant gain over own history (H=3 PR-AUC +0.024 [-0.015, 0.059]); adding country-wide counts is
    (H=3 PR-AUC +0.015 [0.003, 0.029], Brier -0.0033 [-0.0060, -0.0007] vs own+nbr; H=6 also). The user's guess about nationwide activity was right, about neighbours it is not proven.
  - The best C was the smallest of the grid (0.01) for every set: the grid is cut at the edge. To extend downward, chosen on validation.
- Caveats: many comparisons were printed (3 horizons, 3 feature sets, 2 references); only H=3 vs the best baseline was the planned comparison, the rest is exploratory.
  Coefficients with C=0.01 are heavily shrunk, a negative coefficient of Sumska is not to be read as a cause.

## 5. Extended grids, Platt recalibration, robustness. RETRACTION of the first H=3 headline
- Grids extended on the edges (C down to 0.001, k up to 6400), chosen on validation as before; Platt recalibration
  (sigmoid fitted on the newest 28 days of the past, model on the older part, gap H) added as separate families.
  Outputs: results/experiment_v2.txt, results/robustness.txt (scripts/run_experiment.py, scripts/run_robustness.py).
- RETRACTION: decision 4 said "H=3: PR-AUC +0.069 [0.018, 0.116] over the best baseline". With the extended grid,
  validation picks C=0.001, 90 d and the same comparison gives +0.033 [-0.008, 0.069]: not distinguishable from zero.
  So the first number depended on which hyperparameters the narrow grid allowed. C=0.001 is again the smallest value
  of the grid, i.e. validation keeps asking for predictions that are almost the constant. Brier on validation barely moves
  (0.2264 vs 0.2256 for the constant).
- What holds:
  - H=1: PR-AUC +0.066 [0.034, 0.095] over the constant (all feature sets positive); stable between the two grids
    (+0.081 before), with naive-affected moments removed (+0.066 [0.031, 0.103]), without the last 24 h (+0.065 [0.035, 0.096]),
    and on a period before the hole of the official data for both sources: volunteer +0.049 [0.022, 0.076],
    official +0.063 [0.018, 0.106].
  - H=3: positive but uncertain: main test +0.033 [-0.008, 0.069]; shifted period volunteer +0.056 [0.016, 0.090],
    official +0.044 [-0.007, 0.087].
  - Brier: no significant improvement over the best baseline in any of these runs (differences -0.005..+0.003, intervals contain 0).
  - Platt recalibration did not help (test Brier unchanged, PR-AUC lower at H=1). The slope of the best model is 0.47 at H=3
    (below 1: predictions more spread than the signal supports). Slopes of the baselines are meaningless (their predictions
    are step-wise constant between blocks) and are not to be read.
  - Neighbours vs own history: no gain; country-wide counts: at most a small one (H=6 Brier -0.0014 [-0.0023, -0.0004] vs own+nbr,
    but H=3 PR-AUC +0.008 [-0.004, 0.020]). The claim "country-wide activity helps at H=3" from decision 4 is withdrawn too.
- Sources differ in base rate (H=3, same period: volunteer 64.8%, official 72.0%), as expected from the different
  splitting of alerts; the best baseline also differs by period (recent_activity in the shifted one). Compare direction, not levels.
- Selection rule is Brier on validation for every family; PR-AUC would have picked other configurations. Not tried, to avoid
  one more round of choices after seeing the test.

## 6. Gradient boosting (time-boxed), criterion fixed before the run
- Model: sklearn HistGradientBoostingClassifier, learning rate 0.05, min 200 rows per leaf, L2 1.0, no early stopping
  (its validation split is random, not time-ordered), grid depth {2, 3} x iterations {50, 150} x window {365, 180, 90} d,
  feature sets own and own+nbr+cty. Horizons H=1 and H=3. Selection on validation by Brier, as for every family.
- The test block has already been seen for the baselines and the logistic regression (decisions 3-5), so it is not untouched.
  For boosting it is looked at once; no second round of choices after it.
- Criterion (written before the first run): boosting counts as better than the logistic regression only if the paired 95% interval
  of the difference is entirely above 0 for PR-AUC or entirely below 0 for Brier, at H=1 or H=3. Otherwise: no gain, reported as such.
- Further option, not done yet: the dataset updates daily, so a later snapshot would give a block that nobody has looked at.

### Result of decision 6 (results/boosting.txt; run twice, identical numbers)
- Criterion, as written before the run: met at H=1, not met at H=3.
  - H=1, hgb[own+nbr+cty] vs the best logistic family chosen on validation (logreg[own+nbr]+platt):
    PR-AUC +0.052 [0.022, 0.085], Brier -0.0029 [-0.0058, -0.0004].
  - H=3, vs logreg[own+nbr+cty]+platt: PR-AUC -0.013 [-0.059, 0.031], Brier +0.0022 [-0.0026, 0.0073]. No gain.
- Caveat found after the first run (post hoc, so to be read with care): the reference chosen by validation was a weak logistic variant.
  Against logreg[own+nbr+cty] without Platt the H=1 difference is PR-AUC +0.015 [-0.013, 0.046], Brier -0.0010 [-0.0035, 0.0012]:
  not distinguishable from zero. Fair summary: boosting is about as good as the best logistic regression, not demonstrably better.
- Post hoc, one of 22 pairs printed, so exploratory: inside boosting the neighbour and country features matter at H=1:
  hgb[own+nbr+cty] vs hgb[own] PR-AUC +0.051 [0.017, 0.088], Brier -0.0034 [-0.0066, -0.0004]. With logistic regression the same
  step was small and not significant (+0.013 [-0.012, 0.040]), so the information probably needs a non-linear model.
- Brier against the constant stays unproven for boosting too (H=1: -0.0022 [-0.0053, 0.0007]).
- Mistake of mine during this step: a string with a line break in the post-hoc block made the script fail and overwrote the
  saved first-run output. The first run had been copied beforehand, so nothing was lost; now the script is compile-checked before long runs.
- Time box closed: no further model work.
