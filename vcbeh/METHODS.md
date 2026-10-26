# Methods

How each measure in this pipeline is defined. Written for someone reading the results;
for the code, start at `src/vcbeh/ana_vavi.py`.

Notation: `ll` is the list length (15 here), and a *list* is one study–test trial.

---

## Design

Each list presents `ll` words, one at a time, each carrying a point value from 1 to 50.
Two list types are interleaved:

| `pres_types` | Condition | Value sequence |
|---|---|---|
| 1 | **Random** | values drawn independently across serial positions |
| 2 | **Autocorrelated** | neighbouring serial positions carry similar values |

After free recall, the subject reports **one** estimate of the whole list's average value
(`est_values`). The true average is `act_values`. The comparison between the two, and its
decomposition into recalled and non-recalled items, is what the pipeline exists to measure.

Every analysis is run three times — over all lists (`both`), random lists only (`rand`), and
autocorrelated lists only (`temp`) — so any effect can be checked against its list type.

**Data sources.** Trials can be built either from the saved `.mat` files (`load_data`) or directly
from the raw event log (`load_raw`). The raw path is preferred: it reproduces the `.mat` files
exactly, matrix for matrix, while avoiding their staleness, and it is the only way to get a
consistent 34-subject dataset.

The raw export is the authoritative source, but the two on-disk forms are not interchangeable —
each lacks what the other has. Only the concatenated CSV carries `itemno_all`, the item numbering
the matrices use (the per-session TSVs have no item number at all on presentation rows); only the
TSVs carry the `experiment` column. So the CSV supplies the data and the TSVs supply the
labelling, joined on subject and session.

**Inclusion.** Session 0 is a 5-list intro session; numbered sessions have 10 lists. Excluding it
also removes any subject who did nothing else — 8 of the 34 — leaving **26 subjects and 1120
lists**. That is the default (`include_session0=False`) and it reproduces `data_050526_xs0.mat`.

## Data representation

`recalls` is one row per list, one column per output position. Entries are **serial
positions** `1..ll`; `0` is padding after the last response and `-1` marks an intrusion.
`rec_itemnos` and `rec_values` align with it column for column; `pres_itemnos` and
`pres_values` are indexed by serial position instead.

All integer fields are widened to `int64` on load. This is not cosmetic — see
[**B2**](#b2-bysess-never-split-by-session-in-matlab).

## Preprocessing

**Clean-recalls mask.** A response counts when it is a real serial position (`> 0`) *and*
that position has not already been output in the same list. Repetitions keep the first
occurrence and drop the rest. Every recall measure below is computed through this mask.

**Intrusion taxonomy** (`intru_types`), each divided by the number of words studied:

| Type | Condition | Meaning |
|---|---|---|
| WLI | `intrusions == -1` and `rec_itemnos > 0` | within-list |
| PLI | `intrusions > 0` | from a prior list (the value gives how many lists back) |
| ELI | `intrusions == -1` and `rec_itemnos == -1` | extra-list |

Intrusions are excluded from the value analyses. A word recalled at a plausible time but
belonging to another list carries no defensible value for *this* list.

---

## Recall measures

### Serial position curve (`spcs`)

For each subject and serial position `p`, the number of cells where a clean recall equals `p`,
divided by the number of lists in which `p` was presented. Undefined (NaN) when the
denominator is zero — which is load-bearing for `prec_byval`, below.

Counting *cells* rather than *lists* coincides with the usual definition only because the
clean mask removes repeats. The distinction matters when a value-filtered mask is passed in.

### Probability of first recall (`pfrs`)

Same denominator, but the numerator counts only **output position 1, literally** — not the
first *valid* recall. A list whose first response is an intrusion therefore contributes to the
denominator and to no numerator, so **rows can sum to less than 1** (observed: 0.78–1.00).
This is a real property of the measure, not a normalization error.

### Recall probability by item value (`prec_byval`)

The SPC restricted to items of a given value, then averaged across serial positions.
Column `v-1` holds value `v` — the matrix is **indexed by value, not by position**, which is
only well defined because the values present are exactly `1..50`. The loader asserts that.

### Lag-CRP (`crps`)

Kahana (1996). For each adjacent pair of clean recalls, the *lag* is the difference in serial
position between them. Column `c` of the result holds lag `c - (ll-1)`; use `lag_index()` and
`lag_axis()` rather than arithmetic on raw column numbers.

The denominator is what makes this conditional. For each scored transition from serial
position `a`, every position that was **presented, not yet output, and not `a` itself** counts
as one opportunity at its own lag. Four details, each verified against the MATLAB output:

- The available set excludes every previously output position, **including the current `from`
  item**, which has not yet been added to the output history.
- The output history is updated for any positive response, **even when the transition was not
  scored** (say its second endpoint was an intrusion) and even when the response was a
  masked-out repeat.
- A transition whose second endpoint fails the mask contributes to **neither** count.
- Counts are pooled across all of a subject's lists and divided **once** — not averaged over
  per-list ratios.

Lag 0 is undefined and always NaN. Figures draw backward and forward lags as separate lines so
nothing is connected across it.

### Temporal clustering factor (`temp_clusts`)

Polyn, Norman & Kahana (2009). For each scored transition, the actual lag is ranked against
the lags of all still-available items: 1.0 means the nearest available item was chosen every
time, 0.5 is chance.

With `gt` candidates further than the chosen one and `eq` at the same distance out of `n`
available:

```
score = (gt + (eq - 1) / 2) / (n - 1)        # midrank
```

Ties use **midrank**. This was determined empirically, not assumed: midrank reproduces the
saved MATLAB values to 2e-16, while both strict `>` and `>=` are off by 0.024. `eq` is always
at least 1 because the chosen transition is itself a candidate, so `(eq-1)/2` is half credit
for the *other* equidistant candidates. Transitions with one or zero candidates are skipped.
Scores are pooled over all transitions of all of a subject's lists and averaged once.

---

## Value measures

All of these are the same operation — a mask-weighted average of one value per list:

```
list value = sum(values * mask) / sum(mask)
```

The mask may be boolean or **weighted**, so the denominator is a sum of weights rather than a
count. NaN values (intrusions) drop out of the numerator while their weight remains in the
denominator. A list whose mask sums to zero is marked `Inf`, deliberately distinct from NaN,
so "no valid items" can be told apart from "this list does not exist".

| Field | Mask | Meaning |
|---|---|---|
| `est_list` | all lists | the subject's reported estimate |
| `act_pres_list` | all presented items | the list's true average value |
| `act_rec_list` | clean recalls | mean value of what was recalled |
| `act_nonrec_list` | presented **and not** recalled | mean value of what was missed |
| `act_nrec_list` | — | how many items were recalled |
| `act_weightrec_list`, `act_wtdscrec_list` | recalls weighted by `max_op - op + 1` | earlier recalls count more |
| `act_wtascrec_list` | see [**B1**](#b1-act_wtascrec_list-is-not-ascending) | intended: later recalls count more |

The non-recalled set is derived by matching recalled item numbers back against the
presentation list, and the two are checked to correspond exactly.

These three fields are tied together by an identity that holds for every partially recalled
list, and which the test suite asserts:

```
(nrec * act_rec_list + (ll - nrec) * act_nonrec_list) / ll  ==  act_pres_list
```

**Output-position weights.** Positions run to the *last response of the list*, so intrusions
still consume positions. That is deliberate in the original: an error still occupies a slot in
the output order.

**A list where everything was recalled** has no non-recalled items, so `act_nonrec_list` is set
to NaN there. A list with **no** recalls is the mirror case: `act_rec_list` is `Inf` while
`act_nonrec_list` is legitimately the average of the whole list. See
[**B5**](#b5-the-inf-alignment-assertion-is-unsatisfiable-when-a-list-has-no-recalls).

---

## Error measures

| Field | Definition |
|---|---|
| `signerr_pres_list` | `est_list - act_pres_list` — sign shows the direction of bias |
| `abserr_pres_list` | `abs(act_pres_list - est_list)` — accuracy against the whole list |
| `abserr_rec_list` | `abs(act_rec_list - est_list)` — accuracy against what was recalled |
| `abserr_wtrec_list` | as above, output-position weighted |
| `baseerr_rec_list` | `abs(act_pres_list - 25)` — the no-information baseline |

`25` is the midpoint of the 1–50 range: the error a subject would make by ignoring the list
and guessing the middle every time. It is the reference any real estimate has to beat.

**Aggregation.** `*_sub` fields average over a subject's lists, skipping NaN.
`mm_*` fields give per-subject minimum and maximum (MATLAB's `min`/`max` skip NaN, so these
are `nanmin`/`nanmax`).

`*_sess` fields average within session, and two properties are easy to misread:

- Columns are session **ordinals**, not session numbers. A subject missing session 3 has their
  session-4 mean land in column 3.
- The average is a **plain** mean, so a session containing any NaN list yields NaN. This is
  the original behaviour and it is what the saved results contain.

---

## Statistics

**List-type contrasts.** **Wilcoxon signed-rank tests** across subjects, random vs
autocorrelated, on estimation error, recall probability and temporal clustering.

The MATLAB original used paired *t*-tests, which was reasonable for the 22-subject `.mat`: there
the paired differences are close to normal (Shapiro-Wilk p = 0.75-0.93). On the raw 26-subject
data they are not (p = 0.0018, 0.0005, 0.0003, with skew from -1.55 to +2.01), so a rank-based
test is the appropriate choice. Switching changes no conclusion -- nothing is significant either
way -- but it moves p-values substantially; recall probability goes from 0.395 to 0.98.

Both tests are reported along with the Shapiro-Wilk p, so the choice of test is auditable rather
than silent. The bar charts show group means with standard errors, as the MATLAB originals did,
with the Wilcoxon p in the title.

One caveat worth recording: the signed-rank test assumes the differences are symmetric for a clean
median-shift reading, and at skew -1.55/+2.01 that is itself questionable. It remains a better fit
than the *t*-test; a sign test would be the next fallback.

**Error–recall correlation.** Pearson correlation between each subject's mean absolute
estimation error and their mean recall probability, with a least-squares line for the figure.

**Loftus–Masson intervals.** Loftus & Masson (1994), *Psychonomic Bulletin & Review*
1(4):476–490. Each subject is normalized to the grand mean before the interval is derived from
the subject-by-condition interaction. This removes between-subject variance, so the intervals
support comparisons **between conditions** — which ordinary standard errors do not, being
inflated by overall differences between subjects.

**Mixed-effects regressions.** The question is whether the estimate is built from recalled
items, non-recalled items, or both. A full account predicts

```
ValueEst ~ (nrec/ll) * ValueRecalled + ((ll-nrec)/ll) * ValueNotRecalled
```

but `nrec` varies list by list, so no single weighting fits all lists at once. The workaround
is one model per recall count, with a random intercept per subject. Lists with `nrec == ll`
are excluded, because there the two accounts coincide.

Two sweeps, matching the two original scripts: an additive model with both terms weighted
(`nrec` 7–13), and an interaction model with the recalled term unweighted (`nrec` 9–14).

Matching MATLAB's `fitlme` requires forcing three settings, since `statsmodels` differs from it
by default on all of them — and each changes the reported numbers:

- **Maximum likelihood**, not REML (`fitlme`'s default is ML).
- **p-values from a *t* distribution on residual degrees of freedom**, not the z-tests
  `statsmodels` reports (`fitlme` defaults to `DFMethod='residual'`).
- **Conditional R²**, including the random effects, which `statsmodels.fittedvalues` excludes.

Each model is fit with several optimizers and the highest-likelihood fit is kept: L-BFGS alone
will sometimes report successful convergence at a degenerate boundary solution (group variance
exactly zero, intercept pinned to zero).

### Bias by list type

The sweeps above pool both list types, so a difference between them cannot show up. Four further
models fit each list type separately — twice over, once on the raw relationship and once with the
recall/non-recall decomposition:

```
uncorrected:  ValueEst ~ ValuePres                          + (1|Subject)
corrected:    ValueEst ~ ValueRecallW + ValueNotRecallW     + (1|Subject)
```

Here each list is weighted by **its own** recall count rather than the models being split by it:

```
ValueRecallW    = (nrec/ll)      * act_rec_list
ValueNotRecallW = ((ll-nrec)/ll) * act_nonrec_list
```

That is what makes this one model per condition instead of seven. The per-`nrec` sweeps exist
because the original MATLAB compared the weighted and unweighted accounts at a fixed `nrec`, where
a single weighting cannot serve both; that constraint does not apply when the question is simply
how much each source contributes. Splitting the sweeps by condition instead would leave 38–58
lists per cell.

The corrected models drop lists where nothing or everything was recalled — `act_nonrec_list` is
undefined for a fully recalled list, and a list with no recalls has no recalled value. The
uncorrected models keep every list, since they do not use `nrec` at all.

**Mean-centring (optional, off by default).** Models are fit with uncentred predictors, matching
how the original analyses were run; `center=True` centres each continuous predictor at its mean,
computed over both conditions pooled so the two fits share one reference point. `ValuePres` and `ValueAll` centre at
25.40; the list-share-weighted `ValueRecallW` and `ValueNotRecallW` centre at 16.67 and 8.68,
since the weighting puts them on a different scale. Centring each at its own mean preserves the
decomposition identity — the centred parts still sum to the centred whole.

This changes nothing about model fit: slopes, interaction terms, fitted values and R-squared are
all unaffected. It changes only the terms whose meaning depends on where zero sits — the intercept
and, where an interaction is present, the other terms in that interaction. Uncentred, those are
evaluated at a list value of **zero**, and observed list averages run from about 8 to 42, so the
uncentred versions are extrapolations well outside the data. Two consequences worth knowing:

- The `ListType` main effect changes sign: −0.726 (p=0.31) uncentred, +0.451 (p=0.075) centred.
  Only the centred one answers "do the conditions differ for a typical list".
- In the `ValueAll x ValueRecalled` model the recalled-value term goes from 0.155 (p=0.036) to
  0.091 (p=0.161) — see below.

Interaction terms are invariant to centring, so every "x list type" test is unaffected.

**Testing a difference.** Two separate fits describe each condition but cannot test whether they
differ. A pooled model per round carries a `ListType` interaction, and those terms are the test:
`ValueNotRecallW:ListType`, for instance, asks whether non-recalled value contributes differently
in autocorrelated lists.

### The ValueAll x ValueRecalled model

An earlier write-up of this analysis reported a different model, fit over every list with one
observation per list per participant:

```
ValueEst ~ 1 + ValueAll * ValueRecalled + (1|Subject)
```

Two ways this differs from the decomposition above, both easy to miss:

- **`ValueRecalled` is the raw mean value of the recalled items** (`act_rec_list`), not the
  list-share-weighted `ValueRecallW`. The two are on different scales and give different
  coefficients.
- **The predictors are collinear, not complementary.** `ValueAll` already includes the recalled
  items, so the recalled term measures what it adds *beyond* the list as a whole. In the
  recalled/non-recalled decomposition the two terms instead sum exactly to `ValueAll`. The split
  of explanatory weight is correspondingly less stable here.

**Provenance.** This model is not the one `linreg_estval_pressrecallv2.m` currently fits. That
script preallocates a `ValuePres` variable at line 14 and never assigns it, while its LaTeX table
block at line 92 labels the coefficients `'Tip, Studied Items'` / `'Tip, Recalled Items'` with
`inds = [1;3;2;4]` -- placing the *studied* term at coefficient 3, which only holds if the formula
was `ValueRecall*ValuePres`. The script was later edited to use `ValueNotRecall`, orphaning
`ValuePres` and leaving the table labels describing a model it no longer fits. Fitting the
recovered formula on `data_050526_45sess.mat` reproduces the reported coefficients exactly
(0.902 / 0.155 / -0.0026 / 0.559), which confirms the reconstruction.

**Centring matters more here than elsewhere.** Because the two predictors interact, the
uncentred coefficient on `ValueRecalled` is its effect where `ValueAll` is **zero** -- a list whose
average value is nothing, which does not occur. On the published dataset the term is 0.155
(p=0.036) uncentred but **0.091 (p=0.161) centred**, with an identical R-squared and an identical
interaction coefficient. It is the same fit seen from a different reference point, and only the
centred version describes a typical list. `center=False` reproduces the published numbers.

**Two R-squareds.** The value reported for that model, 0.84, is the **marginal** R-squared --
fixed effects only, 0.8366. The **conditional** R-squared, which includes the subject random
intercepts and is what MATLAB's `Rsquared.Ordinary` reports, is 0.831. Both are returned, as
`R2_marginal` and `R2`.

**Session 0 breaks this model.** Fit on all 34 subjects with session 0 included, it degenerates:
intercept 18.5 and a *negative* R-squared. A further reason for the default exclusion.

---

## Known issues

### B1: `act_wtascrec_list` is not ascending

`ana_vavi.m:138` builds the "ascending" output-position weight from the descending weight
written on the line immediately above, rather than from the plain recall mask:

```matlab
wtd_rec_mask_dsc(tri,1:max_op) = (max_op:-1:1).*wtd_rec_mask_dsc(tri,1:max_op);  % 137
wtd_rec_mask_asc(tri,1:max_op) = (1:max_op).*wtd_rec_mask_dsc(tri,1:max_op);     % 138
```

The result is a symmetric tent, `op*(max_op-op+1)`, not `op`. Every saved result was computed
this way, so the port reproduces it by default. `ana_vavi(trials, ascending_weight_bug=False)`
gives the intended weight and changes the field by up to **7.97** value units.

**This affects a published number if that field has been reported.** Which one is correct is a
research decision, not a code decision.

### B2: `bysess` never split by session in MATLAB

`ana_vavi.m:20` builds per-session subject identifiers as `subject + 1000*session`. But
`data.session` is `uint8`, so `1000*session` **saturates at 255** for every session 1–5: each
one receives the same offset and the split silently never happened.

This port casts to `int64` at load, so `bysess=True` does what it says — and therefore
**diverges from MATLAB**. There is no saved result to validate it against.

The same class of defect is at `beh_report_compare.m:6`: `uint8(0.1*session)` is `0` for
sessions 1–4 and `1` for session 5, so `subject + .1*session` collapses four sessions together
and makes the fifth collide with the next subject.

### B3: `stats_vaviv2.m` cannot run

It prints a correlation against `prec_sub` at line 31, but that variable is only assigned
inside a block commented out at line 25. The port computes the quantity explicitly.

### B4: the saved `rand`/`temp` weighted values are stale

In both `res_*.mat` files, the output-position-weighted fields under `rand` and `temp` were
produced by an earlier `ana_vavi` that read `max_op` from the **unsubset** recalls matrix,
misaligning every row against its mask. The `both` condition was unaffected, since there the
subset is the full dataset.

The current `ana_vavi.m` no longer has this defect — the saved files simply predate the fix.
The test suite reproduces the misalignment and recovers the saved numbers bit-for-bit, which is
what identifies them as stale rather than as an error in this port.

### B5: the Inf-alignment assertion is unsatisfiable when a list has no recalls

`ana_vavi.m:186` requires `act_rec_list` and `act_nonrec_list` to carry `Inf` in exactly the
same places. The two mean opposite things — no items recalled, versus every item recalled — so
for any non-empty list they can never coincide. Full-recall lists have already been set to NaN
by then, so the assertion only holds when **no list has zero recalls**.

`data_050526_45sess.mat` happens to contain none, so the original passes. `data_050526_xs0.mat`
contains one, and the MATLAB code drops into the debugger on it. This port asserts what is
actually required: no `Inf` survives in `act_nonrec_list`.

### The `ws0` dataset cannot be processed

In `data_050526_ws0.mat`, `recalls` and `rec_itemnos` are 1285 × **65** while `rec_values` is
1285 × **60**. Masks built from the recall matrix cannot index the values matrix, so the value
analyses cannot run at all. This is a data problem, not a porting problem. `load_data` reports
it and `ana_vavi` raises.

That file also includes session 0 (which raises the per-subject list cap from 50 to 55) and
subjects with 5 or 15 lists, who trip the session-count check at `ana_vavi.m:171`.

**Rebuilding from raw fixes it.** The longest trial in the whole dataset has 65 responses, which
is where the 65-column `recalls` came from; whatever produced `ws0.mat` sized `rec_values` at 60
instead. `load_raw` allocates every recall matrix at one width and asserts they agree, so the
mismatch cannot recur — use it rather than `data_050526_ws0.mat` for the 34-subject set.

### `keyboard` statements

The original dropped into the debugger at five points where an invariant failed, which is
unusable non-interactively. Each is now a named function in `src/vcbeh/checks.py` raising
`ParityError` with the offending subject or list.
