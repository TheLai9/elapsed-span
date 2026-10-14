from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def canonical_onsets(values: Sequence[float]) -> NDArray[np.float64]:
    onsets = np.asarray(values, dtype=np.float64)
    if onsets.ndim != 1:
        raise ValueError("Onsets must be one-dimensional.")
    if not np.all(np.isfinite(onsets)):
        raise ValueError("Onsets must be finite.")
    if np.any(np.diff(onsets) < 0):
        raise ValueError("Onsets must be nondecreasing.")
    if onsets.size == 0:
        return np.ascontiguousarray(onsets)
    keep = np.r_[True, np.diff(onsets) != 0]
    return np.ascontiguousarray(onsets[keep])


def normalize_onsets(onsets: NDArray[np.float64]) -> NDArray[np.float64]:
    if onsets.size < 2:
        raise ValueError("At least two distinct onsets are required.")
    elapsed = onsets - onsets[0]
    if not np.all(np.isfinite(elapsed)) or elapsed[-1] <= 0:
        raise ValueError("Elapsed times must have a finite positive span.")
    normalized = np.ascontiguousarray(elapsed / elapsed[-1])
    if not np.all(np.diff(normalized) > 0):
        raise ValueError("Onset intervals exceed floating-point precision.")
    return normalized


def validate_parameters(gap: float, max_step: int) -> None:
    if not np.isfinite(gap) or gap < 0:
        raise ValueError("Gap must be finite and nonnegative.")
    if isinstance(max_step, bool) or not isinstance(max_step, Integral):
        raise TypeError("Max step must be a positive integer.")
    if max_step < 1:
        raise ValueError("Max step must be a positive integer.")
