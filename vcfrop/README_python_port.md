# vcfrop — Python port

Python port of the MATLAB analysis pipeline for the VaVi / e2frop free-recall
value-estimation experiment. Covers the analysis core and the statistics layer;
the figure scripts (`plot_res_forpres_e2frop.m`, `beh_report_compare.m`) were
out of scope and remain MATLAB-only.

Requires numpy, scipy, pandas and statsmodels.

## Usage

```python
from vcfrop import ana_vavi_e2frop, run_stats, run_regression

res = ana_vavi_e2frop("data_083026_xs0.mat")
run_stats(res)
run_regression(res, latex_path="linreg_estval.tex")
```

## Data sources

Two, and they agree exactly.

```python
from vcfrop import load_raw, load_data

data = load_raw("../../vcfrop_modified_data")   # raw BIDS event log (primary)
data = load_data("data_083026_xs0.mat")         # the saved MATLAB matrices
```

`vcfrop/rawio.py` builds the trial matrices directly from the raw event log in
`vcfrop_modified_data/` — one BIDS-style TSV per subject-session, `task-vcfrop`. Excluding
session 0 (a shorter practice session, which also drops LTP663, who has no other) gives
**10 subjects and 470 lists**, the `xs0` convention used throughout.

**The loader is validated against the saved `.mat`**, which the MATLAB pipeline produced from
this same log: 10 analysis fields, 71,440 cells, **0 mismatches**. Two things differ by
design:

- The `.mat`'s `trial` column was only filled in for session 1 and is `0` everywhere else.
  The loader numbers every trial. Nothing in the analysis reads `trial`, so this is a latent
  defect in the saved file rather than a live one.
- `itemno` is `n/a` on presentation rows in the TSVs, so the loader derives item numbers from
  the word strings. They do not match the `.mat`'s numbering, which came from a wordpool file
  that is not in this tree; they are consistent *within* a trial
  (`rec_itemnos[i] == pres_itemnos[recalls[i] - 1]` for all 3092 clean recalls), which is all
  the non-recalled mask needs.

### The hand-curation

`modifiedtrials.xlsx` logs 24 trials where the recall record was ambiguous and was corrected
by hand — a value said twice, a blank where a value should be, a word and its value said out
of order. The paradigm attaches each value to the recall it follows, so a stray utterance
shifts every later pairing in that trial.

Four sessions kept an `unmod-` backup, so the effect can be measured on them: **5 of those 40
trials changed**, four being a single misplaced estimate and one a full re-segmentation of the
recall sequence (`sub 666, session 1, trial 8`). The other logged sessions were edited in
place, so that is a lower bound. `load_raw(..., modified=False)` reads the backups.

The corrected data is what `data_083026_xs0.mat` already contains — the curation predates it —
so running on the raw log reproduces every earlier result rather than changing any.

**One estimate lies outside the 1–50 value range** (a 99, among 2952 scored items). It is left
in place; dropping data is the experimenter's call, not the pipeline's.

## Value-estimation curves (new analyses)

`vcfrop/valuecurves.py` adds estimation-error analogs of the three recall curves. These have
no MATLAB counterpart and no vcbeh counterpart — they are new.

At recall the subject values each item as they recall it, so every recalled item carries both
a true value and an estimate (95.5% of clean recalls here). Values are z-scored within each
list, on that list's own presented values:

```
mu, sd   = mean and sd of the list's 15 presented values
z_act(i) = (true value of recalled item i - mu) / sd
z_est(i) = (subject's estimate of it     - mu) / sd
signed   = z_est - z_act      # positive = overestimate
absolute = |z_est - z_act|
```

Centring on the list's own mean puts every list on a common scale **without** flattening the
bias — a list-wide over- or under-estimate survives the transform, which is the point.

| Function | Recall analog | Bins |
|---|---|---|
| `error_by_serial_position` | SPC | serial position 1..15 |
| `error_by_lag` | lag-CRP | lag −14..+14, lag 0 NaN |
| `error_by_output_position` | PFR (output position 1) | output position 1..20 |

Aggregation is **lists within a session, then sessions within a subject, then subjects**,
unweighted at each level — not a pooled mean. A session in which more items were recalled
does not thereby count for more. `observation_counts` gives the per-bin observation count,
which the notebook uses to truncate each curve where the data run out; a hierarchical mean
makes a bin of three observations look exactly like a bin of three hundred.

`error_by_lag` uses the same adjacency rule as `crp` and `temp_fact` — an intrusion or a
repeat breaks the pair — so the three are defined over the same transitions.

## Ported from vcbeh

`vcfrop/linreg.py` and the lower half of `vcfrop/stats.py` bring over the analyses from
`../vcbeh/notebooks/vcbeh_analysis.ipynb`: the recalled / non-recalled decomposition, the
per-condition models and their `ListType` contrasts, the `ValueAll x ValueRecalled` model,
and the Wilcoxon-based list-type contrasts.

**Validated against the source.** Both implementations were run on vcbeh's own
`data_050526_45sess.mat`: estimates, standard errors, *t* and *p* agree to **0.00e+00**
across every model and every option (corrected/uncorrected, centred/uncentred, pooled/split).

Three things deliberately differ:

1. **R-squared.** vcbeh derives both R² values from `statsmodels`' `fittedvalues`, which in
   statsmodels 0.14.5 **already contains the random effects**; adding the BLUPs to it
   double-counts them. On vcbeh's own data that yields conditional 0.8311 < marginal 0.8366,
   which is impossible, and the two labels are in effect swapped. This port derives both from
   `model.exog`, so the ordering holds by construction and the result does not depend on what
   that property happens to contain.
2. **The sweep range.** vcbeh fits one model per recall count over a hardcoded `nrec` 7–13.
   On 470 lists those cells hold 63, 56, 35, 25, 13, 7, 8 lists, so the upper half cannot
   support a random-intercept fit. `viable_recall_counts` derives the range from the data
   (cells with ≥ 30 lists) and `recall_count_sizes` reports every cell so the choice is
   auditable.
3. **Sample size.** vcbeh's sections 7–8 run on 22–26 subjects and 1080–1120 lists;
   the e2frop cohort is 10 subjects and 470 lists. The analyses carry over; the statistical
   power does not.

### The decomposition models require corrected weights

`ValueRecallW + ValueNotRecallW` must reconstruct `ValuePres` exactly, and that rests on
`act_nrec_list`. That field is built through the output-position weighting, so under the
default MATLAB-faithful path its rows misalign for `rand`/`temp` and 223 of 462 lists fail
the identity. `vcfrop.linreg` raises `InconsistentRecallCountError` rather than fitting on it:

```python
res = ana_vavi_e2frop(path, fix_op_weights=True)   # required for vcfrop.linreg
```

The `both` condition is unaffected, since nothing is filtered out of it.

## Notebook

`notebooks/vcfrop_analysis.ipynb` runs the whole pipeline end to end in 12 sections: dataset,
analysis, parity check, recall measures (including recall probability by item value), value
estimation, the statistics, the published mixed model, the recalled/non-recalled
decomposition, bias by list type, the new value-estimation curves, slide-style summary
tables, and what the weighting bugs change. It orchestrates only; every number comes from
`vcfrop/`. It is committed with its outputs executed (65 cells, 14 figures).

```
jupyter lab notebooks/vcfrop_analysis.ipynb
# or re-execute headless:
python -m nbconvert --to notebook --execute --inplace notebooks/vcfrop_analysis.ipynb
```

Its figures are the one place the port draws anything; the MATLAB figure *scripts*
(`plot_res_forpres_e2frop.m`, `beh_report_compare.m`) remain unported, and there is no
`vcfrop.plots` module. The two series use the project's own colours, with the red stepped
one shade lighter than the MATLAB default (`#ce1a18`, from the red ramp in
`beh_report_compare.m`) so the pair clears a colour-vision check: CVD separation dE 27.1
(protan), normal-vision dE 38.2, both above 3:1 contrast on white.

`loftus_masson` in `vcfrop/stats.py` is a direct translation of `../vcbeh/LoftusMasson.m`
(Loftus & Masson 1994) for the within-subject error bars; it agrees exactly with the
independent implementation in `../vcbeh/src/vcbeh/stats/`.

From the command line:

```
python scripts/run_analysis.py data_083026_xs0.mat res_083026_xs0.mat   # verify
python scripts/run_stats.py    data_083026_xs0.mat --latex linreg_estval.tex
python scripts/reverse_value_from_mask.py                               # phase-0 gate
python -m pytest tests/
```

`ana_vavi_e2frop(datapath, bysess=False, fix_op_weights=False)` returns
`{"unilist", "both", "rand", "temp"}` with the same field names as the MATLAB
struct.

## Mapping

| MATLAB | Python |
|---|---|
| `ana_vavi_e2frop.m` | `vcfrop/analysis.py` |
| `stats_vaviv2.m` (stats only) | `vcfrop/stats.py` |
| `linreg_estval_presrecallval.m` | `vcfrop/regression.py` |
| EMBAM `spc`/`pfr`/`crp`/`temp_fact` | `vcfrop/recall.py` |
| EMBAM `make_clean_recalls_mask2d`/`apply_by_index` | `vcfrop/masks.py` |
| `value_from_mask`/`mean_abserr_sess` | `vcfrop/value.py` |
| raw BIDS event log | `vcfrop/rawio.py` |
| `../vcbeh` `stats/linreg.py` | `vcfrop/linreg.py` |
| `../vcbeh` `stats/contrasts.py` | `vcfrop/stats.py` (lower half) |
| — (new analyses) | `vcfrop/valuecurves.py` |
| — (figures) | `vcfrop/plots.py` |

## Relationship to `../vcbeh`

The sibling directory `../vcbeh` holds the earlier-experiment version of this
pipeline **and an existing Python port of it** (`vcbeh/src/vcbeh/`, with an
`embam/` subpackage). This port was written independently and then
cross-checked against it; see Validation. If the two are to be maintained
together, folding `vcfrop` in as a variant of `vcbeh` is worth considering
rather than keeping two packages — the EMBAM primitives are duplicated.

`../vcbeh` also holds the real `value_from_mask.m` and `mean_abserr_sess.m`.
The EMBAM primitives (`spc`, `pfr`, `crp`, `temp_fact`, `apply_by_index`,
`make_clean_recalls_mask2d`) are still not on this machine in MATLAB form.

## Validation

Every primitive was reimplemented and pinned against the saved
`res_083026_xs0.mat`, then checked a second way:

- **`value_from_mask` and `mean_abserr_sess`** were first recovered empirically
  from the saved result, then confirmed line-by-line against the real MATLAB
  sources in `../vcbeh/`. The recovered definitions were exactly right.
- **`spc`, `pfr`, `crp`, `temp_fact`, `make_clean_recalls_mask2d`** have no
  MATLAB source available, but the independent Python port in
  `../vcbeh/src/vcbeh/embam/` agrees with this one to machine precision
  (max difference 3.3e-16) on all four measures over this dataset. Two
  independently derived implementations converging on the same MATLAB
  reference is the strongest evidence available without MATLAB.

All 69 comparable fields across all three conditions (`both`, `rand`, `temp`)
reproduce the saved result to within 1e-9 relative; most are bit-exact, and the
largest deviation anywhere is 7.1e-15, i.e. floating-point noise.
`act_nrec_list` and `act_nonrec_list` have no reference — the saved file
predates them.

Definitions that had to be recovered, and the evidence that pinned each:

- **`value_from_mask`** — `sum(values.*mask,"omitnan")/sum(mask)`, `Inf` when
  the mask is empty. A NaN *inside* the mask contributes 0 to the numerator but
  still counts in the denominator (trial 8: `92/4 = 23.0`, where a nanmean
  would give 30.67). Derived in `scripts/reverse_value_from_mask.py`, then
  confirmed against `../vcbeh/value_from_mask.m`.
- **`mean_abserr_sess`** — a plain mean, not NaN-omitting, so one undefined
  list makes the whole session NaN (subject 0, session 4). Columns are
  **positional per subject**: column `j` is that subject's j-th distinct
  session, so a subject missing a session has its later sessions shifted left.
  Confirmed against `../vcbeh/mean_abserr_sess.m`.
- **`pfr`** — output position 1 only. A list beginning with an intrusion counts
  in the denominator but credits no serial position. Independently confirmed by
  the `vcbeh` port's own docstring.
- **`temp_fact`** — transitions between *adjacent* output positions, as in
  `crp`; an intrusion or repeat breaks the pair rather than being skipped over.
  Scores pooled over all of a subject's transitions, tied absolute lags split
  evenly.

## Known defects in the MATLAB source

Reproduced by default so results stay comparable. Pass `fix_op_weights=True`
to correct the first two.

1. **Row misalignment** (`ana_vavi_e2frop.m:110-120`) — the weighting loop runs
   over the filtered row count but indexes `data.recalls(tri,:)` on the
   unfiltered array, so `max_op` is read from an unrelated trial. Inert for
   `both`; shifts ~90% of cells for `rand`/`temp`.
   **This is a regression.** The sibling `../vcbeh/ana_vavi.m:136` reads
   `recalls_inds`, the filtered array, i.e. it already has the correct form.
   `fix_op_weights=True` restores that behavior.
2. **Ascending weights are not ascending** (`:116`) — `wtd_rec_mask_asc` is
   built from `wtd_rec_mask_dsc` one line after that array was overwritten,
   giving `(1:n) .* (n:-1:1)`, a symmetric bow rather than a ramp. Affects
   every condition. Present in `../vcbeh/ana_vavi.m:138` too, so it is
   long-standing rather than specific to this file.
3. **`est_rec_list` keeps its Inf sentinel** (`:165-169`) — it is left out of
   the Inf→NaN conversion applied to its five siblings. Faithfully reproduced,
   and confirmed present in the saved MATLAB result. Nothing downstream reads
   the field, but a `nanmean` over it returns Inf rather than skipping the
   no-recall list.
4. **`res.act_nonrec_list`** (`:132`) — assigns into a field that is never
   otherwise created, producing a stray column-major array. Still reproduced, under the name
   **`act_nonrec_stray`**; the name `act_nonrec_list` now holds the real measure (the mean
   value of the non-recalled items), which that MATLAB line was evidently reaching for but
   never computed.
5. `keyboard` at `:142`, `:171`, `:175` was an assertion; these now raise
   `AnalysisConsistencyError`.
6. The LaTeX writer never emitted `\end{tabular}`/`\end{table}`, leaving an
   uncompilable file. The port closes both.
7. **`bysess=True` does something MATLAB never did.** `ana_vavi_e2frop.m:29` builds
   per-session subject ids as `subject + 1000*session`, but `data.session` is `uint8`, so
   `1000*session` **saturates at 255** and every session got the same offset — the split
   silently never happened. This port loads everything as float64, so it does what it says.
   There is no saved result to validate it against.

### Do the weighting bugs change any result?

**No published result changes.** The fitted model,
`ValueEst ~ 1 + ValueRecall*ValuePres + (1|Subject)`, uses `act_rec_list` and
`act_pres_list` only — never the weighted columns — so its coefficients are
identical either way. The bugs reach only `act_weightrec_list` and
`act_wtascrec_list`, which feed the commented-out alternatives
(`fitlmtbl5`, `fitlmtbl7`). Even there the effect is small: in `fitlmtbl5` the
`WeValueRecall` coefficient moves 0.112 → 0.101 (p 0.32 → 0.37), changing no
conclusion.

## MATLAB/statsmodels equivalence notes

- `std` uses `ddof=1` throughout, matching MATLAB's N-1 default.
- `fitlme` defaults to `FitMethod='ML'`; the port passes `reml=False`.
- `fitlme` reports t against residual df (`DFMethod='residual'`); `MixedLM`
  reports z. p-values are recomputed as two-sided t with `nobs - n_fixed` df.
  That reproduces MATLAB's default, which is itself an approximation.
- 469 of 470 lists enter the model; the one all-intrusion list has no
  `ValueRecall` and is dropped, as in MATLAB.
