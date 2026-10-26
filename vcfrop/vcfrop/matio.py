"""Loading of the experiment .mat files.

The source .mat files mix uint8/uint16/int16 across fields. numpy, unlike MATLAB,
does not promote on arithmetic, so ``pres_values - est_values`` on unsigned types
would wrap silently. Everything is therefore normalized to float64 on load.
"""

import numpy as np
import scipy.io as sio

# Fields that are one value per trial. MATLAB keeps them as column vectors and
# the analysis code relies on that shape (e.g. est_values is fed to
# value_from_mask as an n-by-1 "matrix").
_COLUMN_FIELDS = (
    "subject",
    "session",
    "pres_types",
    "est_values",
    "act_values",
    "trial",
)


class Data(dict):
    """Trial data, with fields reachable as attributes (``data.recalls``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self):
        return Data(self)


def _to_2d(arr, as_column):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1) if as_column else arr.reshape(1, -1)
    return arr


def load_data(path):
    """Load a ``data_*.mat`` file into a :class:`Data` mapping.

    ``list_length`` comes back as a plain int; every other field is a float64
    2-D array.
    """
    mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    raw = mat["data"]

    data = Data()
    for name in raw._fieldnames:
        value = getattr(raw, name)
        if name == "list_length":
            data[name] = int(np.asarray(value).ravel()[0])
        else:
            data[name] = _to_2d(value, as_column=name in _COLUMN_FIELDS)
    return data


def load_res(path):
    """Load a saved ``res_*.mat`` into nested dicts.

    Returns ``{"unilist": array, "both": {...}, "rand": {...}, "temp": {...}}``.
    Per-subject vectors (``temp_clusts``, ``abserr_pres_sub``, ...) are kept 1-D,
    matching what the analysis produces.
    """
    mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    raw = mat["res"]

    out = {}
    for name in raw._fieldnames:
        value = getattr(raw, name)
        if name == "unilist":
            out[name] = np.asarray(value, dtype=np.float64).ravel()
        else:
            out[name] = {
                f: np.asarray(getattr(value, f), dtype=np.float64)
                for f in value._fieldnames
            }
    return out
