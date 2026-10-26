"""Python reimplementation of the EMBAM behavioral toolbox functions used here.

This package is a hard boundary: it contains no project-specific logic and can
be lifted out on its own.  The MATLAB originals are from the Kahana lab's EMBAM
toolbox, which is not distributed with this project; every function here was
verified to reproduce the saved MATLAB outputs (``spc``, ``pfr`` and ``crp``
exactly, ``temp_fact`` to ~1e-16).
"""

from .clustering import temp_fact
from .crp import crp, lag_axis, lag_index
from .groupby import group_apply
from .masks import make_clean_recalls_mask2d
from .recall_prob import pfr, spc

__all__ = [
    "crp",
    "group_apply",
    "lag_axis",
    "lag_index",
    "make_clean_recalls_mask2d",
    "pfr",
    "spc",
    "temp_fact",
]
