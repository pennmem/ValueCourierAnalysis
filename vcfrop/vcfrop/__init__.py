"""Python port of the vcfrop (VaVi / e2frop) free-recall value-estimation analyses.

    from vcfrop import ana_vavi_e2frop, run_stats, run_regression

    res = ana_vavi_e2frop("data_083026_xs0.mat")
    run_stats(res)
    run_regression(res, latex_path="linreg_estval.tex")

The recalled / non-recalled decomposition models in :mod:`vcfrop.linreg`
require ``fix_op_weights=True``; see that module's
``InconsistentRecallCountError``.
"""

from . import linreg, plots, rawio, valuecurves
from .analysis import (
    VALUE_BASELINE,
    AnalysisConsistencyError,
    ana_vavi_e2frop,
)
from .masks import apply_by_index, make_clean_recalls_mask2d
from .matio import load_data, load_res
from .rawio import load_raw
from .recall import crp, pfr, spc, temp_fact
from .regression import build_table, run_regression
from .stats import (
    descriptives,
    error_recall_correlation,
    format_report,
    list_type_contrasts,
    loftus_masson,
    recall_probability,
    run_stats,
)
from .value import mean_abserr_sess, value_from_mask
from .valuecurves import (
    error_by_lag,
    error_by_output_position,
    error_by_serial_position,
    item_z_scores,
    observation_counts,
    value_curves,
)

__all__ = [
    "AnalysisConsistencyError",
    "VALUE_BASELINE",
    "ana_vavi_e2frop",
    "apply_by_index",
    "build_table",
    "crp",
    "descriptives",
    "error_by_lag",
    "error_by_output_position",
    "error_by_serial_position",
    "error_recall_correlation",
    "format_report",
    "item_z_scores",
    "linreg",
    "list_type_contrasts",
    "load_data",
    "load_raw",
    "load_res",
    "loftus_masson",
    "make_clean_recalls_mask2d",
    "mean_abserr_sess",
    "observation_counts",
    "pfr",
    "plots",
    "rawio",
    "recall_probability",
    "run_regression",
    "run_stats",
    "spc",
    "temp_fact",
    "value_curves",
    "value_from_mask",
    "valuecurves",
]
