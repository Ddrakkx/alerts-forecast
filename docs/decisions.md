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
