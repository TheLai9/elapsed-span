import numpy as np
from numba import njit

from .onsets import canonical_onsets, normalize_onsets, validate_parameters

SHIFTS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])


@njit(cache=True)
def _fixed_distance(query, reference, gap, max_step, shift, unilateral):
    query_count = len(query)
    reference_count = len(reference)
    costs = np.empty((query_count, reference_count))
    best = np.inf

    for i in range(query_count):
        for j in range(reference_count):
            value = gap * (i + j)
            for a in range(1, min(i, max_step) + 1):
                for b in range(1, min(j, max_step) + 1):
                    if unilateral and a > 1 and b > 1:
                        continue
                    ratio = (query[i] - query[i - a]) / (
                        reference[j] - reference[j - b]
                    )
                    cost = min(abs(np.log2(ratio) + shift), 2.0) * (a + b) / 2 + gap * (
                        a + b - 2
                    )
                    value = min(value, costs[i - a, j - b] + cost)
            costs[i, j] = value
            best = min(
                best,
                value + gap * ((query_count - 1 - i) + (reference_count - 1 - j)),
            )

    return best / max((query_count + reference_count - 2) / 2, 1.0)


@njit(cache=True)
def _profile_distance(query, reference, gap, max_step, unilateral):
    best = np.inf
    for shift in SHIFTS:
        best = min(
            best,
            _fixed_distance(query, reference, gap, max_step, shift, unilateral)
            + 0.05 * abs(shift),
        )
    return best


@njit(cache=True)
def _anchor_prefix_scores(query, reference, gap, max_step, shift):
    query_count = len(query)
    reference_count = len(reference)
    costs = np.empty((query_count, reference_count))
    scores = np.full(reference_count, np.inf)

    for i in range(query_count):
        for j in range(reference_count):
            best = gap * (i + j)
            for a in range(1, min(i, max_step) + 1):
                for b in range(1, min(j, max_step) + 1):
                    residual = (
                        np.log2(
                            (query[i] - query[i - a])
                            / (reference[j] - reference[j - b])
                        )
                        + shift
                    )
                    edge = min(abs(residual), 2.0) * (a + b) / 2 + gap * (a + b - 2)
                    best = min(best, costs[i - a, j - b] + edge)
            costs[i, j] = best

    running = np.inf
    for j in range(reference_count):
        column = np.inf
        for i in range(query_count):
            column = min(
                column,
                costs[i, j] + gap * (query_count - 1 - i) - gap * j,
            )
        running = min(running, column)
        if j >= 1:
            scores[j] = (running + gap * j) / max((query_count + j - 1) / 2, 1.0)

    return scores


def distance(
    query,
    reference,
    *,
    gap: float = 1.0,
    max_step: int = 3,
    family: str = "bilateral",
) -> float:
    validate_parameters(gap, max_step)
    if family not in ("bilateral", "unilateral"):
        raise ValueError("family must be 'bilateral' or 'unilateral'")
    query = canonical_onsets(query)
    reference = canonical_onsets(reference)
    if len(query) < 2 or len(reference) < 2:
        return float("inf")
    return float(
        _profile_distance(
            normalize_onsets(query),
            normalize_onsets(reference),
            gap,
            max_step,
            family == "unilateral",
        )
    )
