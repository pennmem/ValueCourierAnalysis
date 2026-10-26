"""Figures, ported from the MATLAB plotting scripts."""

from .condition_models import (
    plot_condition_coefficients,
    plot_estimate_vs_actual,
    plot_partial_effects,
)
from .pres_values import plot_pres_values
from .res_datasan import plot_crp_by_subject, plot_curves_by_subject
from .res_forpres import (
    plot_error_vs_recall,
    plot_lag_crp,
    plot_pfr,
    plot_recall_probability_bars,
    plot_spc,
    plot_temporal_clustering,
    save_all,
)
from .style import RAND_COLOR, TEMP_COLOR, publication_style
from .sub_sess import plot_session_counts

__all__ = [
    "RAND_COLOR",
    "TEMP_COLOR",
    "plot_condition_coefficients",
    "plot_crp_by_subject",
    "plot_curves_by_subject",
    "plot_error_vs_recall",
    "plot_estimate_vs_actual",
    "plot_lag_crp",
    "plot_partial_effects",
    "plot_pfr",
    "plot_pres_values",
    "plot_recall_probability_bars",
    "plot_session_counts",
    "plot_spc",
    "plot_temporal_clustering",
    "publication_style",
    "save_all",
]
