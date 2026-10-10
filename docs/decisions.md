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
- Results (test, last 8 weeks; scripts/run_experiment.py, full output in results/experiment_v1_narrow_grid_superseded.txt):
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

## 7. More oblasts, and a second post-hoc baseline (written before the re-run)
- Oblasts, each run completely separately (own neighbour list, own hyperparameters chosen on its own validation block, own
  bar, own bootstrap): Poltavska (the main one), Kyivska, Kharkivska, Lvivska. `scripts/run_region.sh "<region>"`, outputs in
  results/<region>/, headline numbers collected by `scripts/summarize_regions.py` into results/summary.md.
- Why Lvivska and not a quieter western oblast: in the last 8 weeks Zakarpatska has 0 alerts (nothing to evaluate), Ivano-Frankivska 8,
  Ternopilska 21, Chernivetska 36. Lvivska has 245 (Poltavska 382), so it is the quietest western one that can be tested.
- Neighbour lists (from my knowledge of the map, NOT from the data; to be checked on a map):
  Kyivska: Zhytomyrska, Chernihivska, Poltavska, Cherkaska, Vinnytska, Kyiv City (an enclave inside the oblast).
  Kharkivska: Sumska, Poltavska, Dnipropetrovska, Donetska (Luhanska is not in the data).
  Lvivska: Volynska, Rivnenska, Ternopilska, Ivano-Frankivska, Zakarpatska. Poltavska unchanged.
- Check after the refactor: Poltavska results are identical to the committed ones (all three result files, compared by sorted content).
- FINDING that changed the plan: in the new oblasts the first runs showed large Brier gains over the bar (Kyivska H=3: -0.049 [-0.064, -0.034]).
  The positive rate there jumps between validation and test (Kyivska H=1: 16% -> 40%; H=3: 39% -> 74%), and every baseline had a static
  training window of at least 90 days (or refit weekly). A diagnostic, written down as post hoc: the constant rate over the last W days refit
  DAILY gives for Kyivska H=3 a test Brier of 0.180 (W=14 d), against 0.1755 for the logistic regression and 0.2243 for the old bar.
  Most of the "gain" in Kyivska and Lvivska was tracking of the changing rate, not predicting onsets. The baselines were weak for such oblasts;
  in the stable Poltavska this did not show.
- Decision: add `recent_level` (constant rate of the last W in {3, 7, 14, 30} days, refit daily) as a second post-hoc baseline for all oblasts,
  Poltavska included; the bar is the best of all baselines on validation. Poltavska numbers may therefore change; whatever comes out is reported.
- Rule kept: the window W is chosen on validation. In Kyivska H=1 validation picked W=90 d (0.1354) and the test then punished that choice
  (0.2674 vs 0.2283 for W=3 d), a choice the validation block could not foresee. I do not change the rule after seeing it; it is a result.
- Safety for quiet oblasts: models fall back to the base rate when a training window has a single class; Platt scaling keeps raw
  probabilities when its calibration part has a single class (tested).

### Result of decision 7 (results/<oblast>/, results/summary.md)
- Poltavska, against the strongest simple baseline (recent level, last 30 days, refit daily): no demonstrated gain for logistic regression or boosting at
  H = 1, 3 or 6 (every paired interval contains zero). The earlier "H=1 gain over the constant, +0.066 [0.034, 0.095]" and "boosting criterion met at H=1" were
  gains over static baselines and are superseded; the README says so.
- Kyivska, Kharkivska, Lvivska, against the best baseline on the test block (strict, chosen with hindsight): PR-AUC gains at H=1 (+0.102, +0.034, +0.041), Brier gains
  at Kyivska H=1 (-0.0168 [-0.0270, -0.0082]), Kharkivska H=1 (-0.0092 [-0.0139, -0.0043]), Lvivska H=3 (-0.0116 [-0.0244, -0.0006]). Out of 16 model-horizon cells (H=1, 3) 5 have a Brier interval excluding zero.
- The validation-chosen bar is fragile under a regime change: in Kyivska it picked a static baseline (recent_activity, Brier 0.2639 at H=1) while the best baseline
  on test was recent_level (0.2351). The strict reference was added after seeing this; it is optimistic for the baselines and is reported next to the usual one.
- The official file cannot serve as a check for Kharkivska (after merging raions the oblast is almost always under alert: 386 moments without alert in the check period
  against about 4,200 in the volunteer data) and Lvivska (0.9% positive against 38%). For Poltavska and Kyivska the direction agrees with the volunteer data.
- Mistakes of mine in this step: the first summary script was written with a string that the shell mangled (a line break inside a quoted string), twice; it stopped a
  command chain before a commit. Two statements in the first README draft were not backed by the data (a count of 6 significant Brier cells instead of 5, a claim about
  Lviv's hour-of-week signal) and one was wrong (the C statement) until I counted; all fixed before the commit.

## 8. Additional short horizons: 15 and 30 minutes (written before the run)
- User's idea: at H=3 the target is close to saturated (Poltava 66% positive moments), so shorter horizons may leave more to predict.
  H = 0.25 h and 0.5 h are ADDITIONAL and exploratory. The main horizon stays H = 3 h; nothing in the main tables is replaced.
- Same protocol, each oblast separately (Poltavska, Kyivska, Kharkivska, Lvivska): weekly walk-forward, purge = H, configurations
  chosen on validation by Brier, two references (best baseline on validation; best baseline on the test block, strict).
- Question 1: do the models beat the references at 15/30 min?
- Question 2 (user's): do neighbours and the country help here? "Help" = paired 95% interval of the difference above 0 for PR-AUC or below 0
  for Brier, for logreg[own+nbr] vs logreg[own], logreg[own+nbr+cty] vs logreg[own], and hgb[own+nbr+cty] vs hgb[own].
- Expectation and caveat written in advance: if neighbours help, it should show most at 15 min, because an incoming wave is often declared
  in neighbouring oblasts first. With a 15-minute horizon part of that is close to nowcasting (the wave is already under way and visible at t).
  That is legitimate (everything is known at t) but the lead time is at most 15 minutes and the README has to say so.
- Bug found before the first run: features were built on the 1-hour grid ("the longest"), but the 15-minute grid reaches 45 minutes further.
  Logistic regression would have crashed on NaN, boosting would have accepted NaN silently. Fixed: features are built on the grid of the
  shortest horizon, and `join_features` refuses any row without features (tested). Main results must stay identical; checked after the run.

### First results of decision 8, and one more correction of the strict reference
- Main results stayed identical after the feature-grid fix (Poltava experiment.txt and the main tables of summary.md compared, equal).
- First run of 15/30 min against the strict reference showed PR-AUC gains in all four oblasts, Poltava included (15 min: +0.021 [0.012, 0.033]).
  But the strict reference had been chosen by BRIER, and the best baseline by PR-AUC can be another one. Diagnostic for Poltava (post hoc):
  against the baseline with the best PR-AUC on the test block (raw hour of week: 0.089 at 15 min, 0.174 at 30 min) the gains are
  +0.008 [-0.006, +0.021] and +0.012 [-0.019, +0.037] for logistic regression, +0.007 / +0.022 (intervals contain 0) for boosting. So the
  Poltava "gain" at short horizons came from comparing PR-AUC with a reference chosen for Brier.
- Rule from now on: the strict reference is chosen separately for each metric (best baseline on the test block by Brier for dBrier, by PR-AUC
  for dPR-AUC). All oblasts and horizons are re-run with it; the summary shows both reference names when they differ.
- Neighbours and country, first reading (comparisons within the same model, not affected by the reference choice): clear help in Kyivska
  (every horizon, largest at 30 min - 1 h) and Kharkivska (15 and 30 min); in Poltavska only for boosting (30 min, 1 h), not for logistic
  regression; in Lvivska none, and at 15/30 min neighbours make the logistic Brier worse (+0.0048 [+0.0009, +0.0102] at 15 min).
  My advance expectation "strongest at 15 min" holds only partly: in Kyivska the gain peaks at 30 min - 1 h.

### Final results of decision 8 (results/<oblast>/*_short.txt, results/summary.md; strict reference per metric)
- Question 1, models vs the strict reference at 15/30 min: Poltavska no (logistic 15 min PR-AUC +0.008 [-0.006, +0.021], Brier -0.0000; 30 min +0.012 [-0.019, +0.037]);
  Kyivska yes, both metrics (30 min: PR-AUC +0.084 [+0.049, +0.118], Brier -0.0075 [-0.0120, -0.0033]); Kharkivska yes, both metrics
  (30 min: +0.053 [+0.018, +0.087], -0.0116 [-0.0162, -0.0070]); Lvivska small PR-AUC gains, and at 15 min the logistic Brier is WORSE
  (+0.0033 [+0.0002, +0.0074]).
- Question 2, neighbours and country (same model with vs without): Kyivska yes at every horizon, Kharkivska yes at 15/30 min, Poltavska only for boosting at
  30 min and 1 h, Lvivska no (logistic Brier worse with neighbours at 15/30 min). Neighbours alone give almost everything: "+nbr" and "+nbr+cty" differ by
  at most 0.009 in PR-AUC for logistic regression.
- The main horizon did not change; the main Poltava conclusion (no demonstrated gain over the strongest baseline) now holds at all five horizons.
- After the per-metric strict reference, the Lviv H = 3 h PR-AUC gain is no longer significant (+0.068 [-0.006, +0.138]); README updated.
- Mistakes in this step: the shell mangled a line break in a scripted edit once more (I switched to direct file edits for anything with backslashes);
  one comparison of mine selected the wrong horizon range and one summary check compared tables of different sizes, both found before drawing conclusions;
  one README sentence said "three times" for four changes.

## 9. Pre-registration of the holdout test on a new snapshot (written 2026-10-10, before any new data is downloaded)
Why: the current test block (2026-08-14..10-09) has been looked at many times and choices were made after seeing it (see decision 10).
The only block nobody has seen is the data that arrives after the current snapshot. Everything below is fixed now.

- Data: a new snapshot of Vadimkin/ukrainian-air-raid-sirens-dataset, volunteer file, downloaded on Monday 2026-10-12; its commit SHA is recorded.
- First step, before any model: compare the two snapshots for alerts that started before the old end (2026-10-09 05:12:41 UTC):
  count added, removed and changed records per oblast. This checks the "late additions" point of the external review. The numbers are reported whatever they are.
- Holdout moments: the 15-minute grid with old end < t <= new end - H, main sample only (no alert active at t). Expected length: about 3 days.
- Selection, exactly as the protocol: validation = the 8 weeks before the holdout start (= the old test block), with a purge of H before the holdout start;
  each family's configuration is chosen by validation Brier. The model is the logistic family with the best validation Brier (`lr_best`);
  the reference is the baseline family with the best validation Brier (bar B, post-hoc baselines included). One training fold:
  rows with t <= holdout start - H; recent_level is refit daily as in the pipeline.
- Primary question (planned): Poltavska, H = 3 h, lr_best vs bar B, PR-AUC and Brier. Prediction from the current results: no gain.
- Secondary questions: Kyivska and Kharkivska at 15 min, 30 min, 1 h: lr_best vs bar B. Prediction: gains (dPR-AUC > 0, dBrier < 0).
  Lvivska is reported with the data warning (decision 10). Also reported, as a hindsight bound only: the best baseline configuration on the holdout.
- Intervals: paired bootstrap over 6-hour blocks of the Kyiv clock (1000 draws, seed 0); day or week blocks are impossible with about 3 days.
  A difference counts only if the 95% interval excludes zero; otherwise the cell is "inconclusive", never "no difference".
  Stated in advance: with about 3 days most cells will probably be inconclusive.
- A cell is "not evaluable" if the holdout has fewer than 10 positive or fewer than 10 negative moments for it.
- Frozen code: the holdout runner and all fixes of decision 10 are committed and tagged `holdout-freeze` before the download.
  After the download the analysis code is not changed; if a bug blocks the run, the fix is a separate commit described as a deviation.
  (Wording of this line corrected on 2026-10-10, before any new data; the first version was garbled, the meaning is unchanged.)

## 10. External review: what was confirmed, and the fixes (2026-10-10)
A second agent reviewed the repository. I re-computed its main claims with my own code before changing anything
(scripts kept outside the repository, the numbers are below). Confirmed:
- Validation was not purged before the test block: its last labels looked into the first H hours of the test block (9-11 rows at H=3),
  and in 2 of 16 cells this changed a chosen configuration (one of them the main logistic model of Kharkivska at 1 h). FIXED: validation ends at test start - H.
- The "strict reference" was the best baseline FAMILY on the test block, with the configuration inside the family chosen on validation, so the README sentence
  "best baseline on the test block itself" was false. Against the best baseline CONFIGURATION on the test block (usually recent_level with W = 3 days)
  Poltavska 3 h gives PR-AUC -0.042 [-0.079, -0.006]. FIXED: the strict reference is now the best of all baseline configurations on the test block, per metric.
  My caveat: a maximum over about 40 configurations chosen on the test block is itself biased towards the baselines, so it is a bound, not the truth;
  the no-hindsight reference (bar B, chosen on validation) is reported next to it.
- The tables showed logreg[own+nbr+cty] without Platt, a variant I picked after seeing results, while the protocol picks the logistic family with the best
  validation Brier (lr_best); they differ in 14 of 16 cells, and e.g. Poltavska 1 h becomes significantly worse in Brier (+0.0013 [+0.0001, +0.0028]).
  FIXED: tables, figures and robustness use the protocol models (lr_best, hgb_best); the neighbour table keeps fixed families on purpose (same model with vs without features).
- Day-block intervals were too narrow: the alert rate persists for weeks. With ISO-week blocks Kyivska 3 h and the Lvivska gains lose significance.
  FIXED: week blocks for all reported intervals (only 9 blocks in 8 weeks, so the intervals are coarse).
- The test block influenced more choices than the README admitted: the reported logistic variant, the choice of Lvivska (by its alert count in the test block),
  and adding 15/30 min after seeing the saturation at 3 h. Now listed in the README.
- Invented ends (naive) also enter neighbour and country features, which the sensitivity run did not cover. In the test block: Poltavska 21 of 382 alerts (5.5%;
  1.9% is the whole-period share), Rivnenska 27 of 107, Ternopilska 9 of 21, Volynska 18 of 216: Lviv's neighbour features rest largely on invented ends. Limitation, not fixed.
- Lvivska: 91% of volunteer alert starts (2026-05..08-28) are not inside any official episode of the oblast (Poltavska 0%, Kyivska 3%; my count, with +-15 min).
  "Different splitting" cannot explain that; the Lviv series describes something the official data does not have. Lvivska stays with a warning (user's decision).
- My README claims that were wrong: "Lviv is the quietest western oblast that can be evaluated" (it is the most active one: Lviv 245, Volyn 216, Rivne 107,
  Khmelnytskyi 75, Chernivtsi 36 alerts in the test block); "in the west alerts are declared over large areas at once" (in the test block a neighbour starts in the
  2 minutes before Lviv's own start in 2% of cases, in 15 minutes in 11%; Poltava 17% and 64%; over the whole period both oblasts are near 35-60%, so the
  claim also depends on the period); "the daily-refit baseline is the conservative direction for the claims" (true for claims of gains, not for the negative Poltava claim);
  "real-data truncation test for more than one oblast" (it ran only for Poltava). FIXED in the README; the real-data truncation test now runs for all four oblasts.
- Not verified by me: late additions to the dataset (needs old snapshots). It is tested on Monday: the holdout run compares the snapshots first (decision 9).
- Not done on purpose: giving the models a 3-day rate feature or daily refits. It would be another change after seeing the test; listed as an extension.
- The holdout runner (scripts/run_holdout.py) and the new-snapshot download (scripts/download_data.py --new-snapshot) were written and dry-run on already seen data
  before any new data exists. One bug found in the dry run's code before freezing: the verdict text called a positive Brier difference "better"; fixed and tested.

### Results after the fixes of decision 10 (results/<oblast>/, results/summary.md; protocol models, week blocks, purge, strict = best baseline configuration)
- Poltavska: against the strict reference nothing at any horizon, for either model (e.g. 3 h logistic PR-AUC -0.045 [-0.079, +0.025]; with week blocks the
  review's "significantly worse", found with day blocks, is no longer significant). Against the no-hindsight reference: small ranking gains at 15 min
  (logistic and boosting PR-AUC +0.021) and for boosting at 30 min (+0.042) and 1 h (+0.035); none at 3 h. The verdict depends on the reference; the README says so.
- Kyivska against the strict reference: PR-AUC gains at 15 min, 30 min, 1 h (+0.052, +0.069, +0.060), Brier gains at 15 and 30 min; nothing at 3 h.
  At 1 h and 3 h it repeats on the earlier period 2026-07-04..08-28 (1 h: PR-AUC +0.057, Brier -0.0055; 3 h: +0.045, -0.0076); on the official source same direction, not significant.
- Kharkivska against the strict reference: both metrics at 15 and 30 min for both models; Brier at 1 h; earlier period at 1 h: PR-AUC +0.062, Brier -0.0101.
- Lvivska: only a tiny PR-AUC gain at 15 min (+0.010); nothing else. With week blocks the earlier "neighbours make Lviv's Brier worse" is no longer significant.
- Neighbours and country (same model with vs without): Kyivska yes at every horizon up to 3 h; Kharkivska mostly in Brier; Poltavska only for boosting (30 min, 1 h);
  Lvivska no.
- Three statements in my first draft of the new README were not backed by the data and were fixed before the commit: the subset checks were claimed for the
  short horizons (they were run at 1 h and 3 h only), the official source was said to agree for Kharkiv (it is unusable there and points the other way), and
  "every Lviv gain disappeared" (one tiny gain remains).

## 11. Interactive demo (demo/), added after the freeze, without touching frozen code
- Request: a visual page that shows how the forecast works on one day in Poltava oblast: alerts on a time axis, the forecast moment t,
  what is known before t and what is hidden after it, the window (t, t+H], and the model's forecast against the baseline.
- Rule kept: nothing in src/ or scripts/ is changed (tag `holdout-freeze`); everything new is in demo/. The demo imports src/ read-only.
- demo/build_demo.py calls the same evaluate() as scripts/run_experiment.py for H = 15 min, 30 min, 1 h, 3 h, so the curves are the real
  out-of-sample walk-forward predictions of the protocol logistic model and of the no-hindsight baseline (bar B) for every 15-minute moment
  of 55 whole days of the test block. Checked: the model and baseline names and configurations equal those in results/poltavska/.
  Lanes for the 7 neighbouring oblasts and a "known at t" panel (minutes since the last own alert, own starts in 24 h, neighbours and
  oblasts under alert) show part of the features.
- Default day: the test-block day with the most alert starts in Poltava (2026-09-23, 15 starts), a plain rule so the page has something to show.
  The page says that the test block has been looked at many times, so it explains the method and is not a new result.
- demo/index.html is one self-contained file (data inlined, about 0.64 MB, no external libraries), built from demo/template.html;
  demo/data.js is a local cache and not committed. Reason: the app's preview pane opened local files as static snapshots and did not load
  a separate data file, which is also how a reviewer might open it.
- Checked in the browser: light and dark theme, desktop and phone width, all 55 days x 4 horizons x reveal on/off x 3 moments
  rendered without a script error (1320 renders).
- Mistakes on the way: a scripted edit failed on an escaped line break (same trap as before) and the next command then recomputed the data
  instead of only rebuilding the page (harmless: the output was byte-identical, which also confirms determinism); the page first stayed
  empty because Python wrote horizon keys as "1.0"/"3.0" while the page looked up "1"/"3"; labels overlapped in the first render. All fixed.

## 12. The live eMap feed and the "yellow" alert level (2026-10-10)
Question from the user: what does alert_level "yellow" in vadimklimenko.com/map/statuses.json mean, is it in our data, and does the feed agree
with our volunteer source? Everything new is in demo/; src/ and scripts/ are untouched (frozen).
- The feed (checked with one request, then the logger): {"version": 10, "states": {oblast: {enabled, enabled_at, disabled_at, alert_level,
  districts: {raion: {...}}}}}, 27 states, 126 raions. Enabled raions have alert_level "red" or "yellow" and an exact enabled_at; disabled
  entries carry no timestamps at all. Luhansk oblast is enabled with "red" since 2022-04-04 16:45 UTC, Crimea and Sevastopol since 2022-12-10
  22:22 UTC: exactly the two permanent sirens described in the dataset README, so the feed and the dataset share their origin.
- Meaning of yellow, from sources, not from memory:
  - the map's own code ranks red 3 > orange 2 > yellow 1, shows a yellow raion as alarmed, and its legend says only "Жовтий рівень тривоги";
  - TSN, 2026-09-02 (tsn.ua/ukrayina/povitriana-tryvoha-v-ukrayini-zminytsia-shcho-oznachatymut-zovtyy-i-chervonyy-rivni-zahrozy-3160946.html):
    the President announced two levels, yellow = a drone (UAV) raid, red = missile / ballistic threats and massed attacks, with different sirens;
  - texty.org.ua, 2025-12-15: raion-level alerting since December 2025; it does not mention levels.
  So yellow exists only since about September 2026, which is INSIDE our test block (2026-08-14..10-09). When it actually started in the
  data is not known (the announcement says "will mean").
- In our data: no alert level anywhere. The volunteer file has region, started_at, finished_at, naive; the official file's "level" column is the
  geographic level (oblast / raion / hromada), not a colour. Neither CSV contains "yellow"/"red". The dataset's processors (pinned snapshot) know no
  levels either. Two traps found in that code: in the official Telegram channel the emoji 🟡 has meant a PARTIAL all-clear ("відбій в області,
  тривога ще триває у якомусь районі"), not the new level; and the official parser counts a start only if the first line has "Повітряна" or 🔴.
  The volunteer parser counts any message with "тривога", "загроза", "небезпека", "сирена" ... as an alert, so drone (yellow) alerts are most
  likely included as ordinary alerts there, but without the messages this is unproven.
- Consequence for the project, a hypothesis only: from September 2026 the target "an alert starts" may mix drone and missile alerts in a new way,
  a possible regime change inside the test block. Not testable with the data we have; noted as a limitation.
- Agreement with the volunteer source can only be checked on a common period, i.e. after 2026-10-09, which is the pre-registered holdout
  (decision 9). To keep that clean, the comparison is done only after the Monday holdout run. The analysis code is frozen anyway, so nothing
  seen in the live feed can change it.
- demo/emap_logger.py: one request per minute at most (a smaller interval is refused), conditional requests (ETag / Last-Modified), an identifying
  User-Agent, doubling pause after errors (up to 10 min). Writes demo/emap_log/ (not committed): events.jsonl (start / end / level change, with the
  poll times around it; an end is only known to lie between two polls), polls.csv, and a full snapshot at start. Tested without network
  (demo/test_emap_logger.py, 5 tests) and with 3 live polls (exactly 60 s apart). Started in the user's terminal on 2026-10-10 ~12:28 UTC;
  it runs only while that terminal (and the computer) runs.
- Mistake found right after starting it: the first poll of the long run came 20 s after the last poll of the 3-poll test run, because each
  process only spaced its own requests. Fixed: at start the logger reads the last poll time from polls.csv and waits until a minute
  has passed (tested); the logger was restarted with the fix.
- Planned after the holdout: aggregate the logged raion events to oblasts (an oblast is "under alert" if any raion is, and separately red only)
  and match them with the new volunteer snapshot for the logged period: which volunteer starts coincide with red and which with yellow raion starts.

## 13. Live forecasts at each all-clear in Poltava oblast (pre-registered 2026-10-10, before the first forecast)
- Trigger: the eMap log (decision 12) shows that the last enabled entry of Poltava oblast (any raion, or the oblast itself) went off.
  The forecast moment t is the logger's poll time at which this was seen (that is when it is known). The forecaster reads only the local log,
  so it sends no extra requests to the server. Only all-clears seen after the forecaster starts count; nothing is back-filled.
- Model and baseline: for each H in {15 min, 30 min, 1 h, 3 h}, configurations chosen on the 8 weeks before the end of our snapshot
  (2026-10-09 05:12:41 UTC, purged by H), exactly as the holdout runner does; the protocol model (logistic family with the best validation Brier)
  and the no-hindsight baseline (bar B). Both are trained once on the volunteer snapshot (training rows t <= snapshot end - H) and not updated live.
- Features at t: the same feature code (src/, frozen) on a history made of the volunteer snapshot up to its end plus eMap raion alerts merged into
  oblast episodes (an oblast is under alert while any of its raions or the oblast itself is, any level). An alert still running at t counts as
  running with an unknown end. Known problems, stated in advance: (1) training on volunteer data, live features from the official feed;
  (2) no data between 2026-10-09 05:12 and the logger start 2026-10-10 12:25 UTC, so counts over 24 h are incomplete until 2026-10-11 12:25
  and over 7 days until 2026-10-17 12:25; every forecast records this status.
- Answers: primary = the volunteer dataset (the model's own target): did a Poltava alert start in (t, t+H]; checked when a newer snapshot exists
  (not before the Monday holdout run). Secondary, available at once: eMap "any" (a Poltava raion or the oblast went on in (t, t+H]) and eMap "red"
  (the same, with level red at the start).
- Each forecast is written to demo/live/forecasts.jsonl with t and the wall-clock time it was made, before its answer can be known. Live forecasts are
  never edited; on Monday they may additionally be recomputed with the gap filled, reported separately.
- What is reported: every forecast with its answers, and per H the Brier score of model and baseline and how often each was closer to the answer.
  Stated in advance: with about 15-25 all-clears before the deadline this is an illustration, not evidence; no significance will be claimed.
- Clarification written before the first forecast: a forecast counts only if it is written at most 3 minutes after the all-clear moment t
  (otherwise an answer could already be in the log); later ones are written with counted = false and left out of the summary. The model training at
  start takes minutes, so an all-clear during training would be late. Code: demo/live_forecast.py (tests: demo/test_live_forecast.py), committed before it ran.
- Bug fixed before any forecast existed: the report decided whether an answer window had passed by the time of the last log EVENT, but quiet minutes
  write no event, so a window without any alert would have stayed "unknown". It now uses the last successful poll in polls.csv (tested).
  The running forecaster did not need a restart: forecasting code was unchanged, the report runs as a separate command.
- 2026-10-10 ~14:48-16:4x UTC the computer was switched off: no polls, no forecasts. First live forecast before that: all-clear at 14:12:29 UTC,
  written 13 s later (counted). Change after it (metadata only, the forecasting code is unchanged): the history status of a forecast now lists every gap
  in the log within the last 7 days (stretches of more than 5 minutes without a successful poll), not only the gap before the logger started.
  An alert that started and ended inside a gap is missing from the history; one that ended inside a gap gets the restart time as its end.
  Restart order: the logger first, the forecaster after its first poll, so the artificial "all-clear" at the logger's restart can never be forecast.
- Report fix (no forecast changed): an answer window that overlaps a gap in the log stays unknown unless a start was seen in it, because alerts that
  started and ended inside the gap are not in the log. For the first forecast the 15 and 30 min answers are known (no alert; model 3% / 7%,
  baseline 4% / 16%), the 1 h and 3 h windows overlap the switch-off and stay open until the volunteer data (which covers the gap) arrive.
