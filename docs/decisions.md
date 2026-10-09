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
