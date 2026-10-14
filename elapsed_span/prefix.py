from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .alignment import SHIFTS, _anchor_prefix_scores, _profile_distance
from .onsets import canonical_onsets, normalize_onsets, validate_parameters


@dataclass(frozen=True)
class PrefixMatch:
    distance: float = float("inf")
    prefix_onsets: int | None = None


def _count_bound(query_count: int, reference_count: int, gap: float) -> float:
    return float(
        gap
        * abs(query_count - reference_count)
        / max((query_count + reference_count - 2) / 2, 1.0)
    )


def _update(match: PrefixMatch, score: float, endpoint: int) -> PrefixMatch:
    # Exact ties select the shortest canonical reference prefix.
    if score < match.distance or (
        score == match.distance
        and (match.prefix_onsets is None or endpoint < match.prefix_onsets)
    ):
        return PrefixMatch(score, endpoint)
    return match


def _prefix_score(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    endpoint: int,
    gap: float,
    max_step: int,
    unilateral: bool,
) -> float:
    return float(
        _profile_distance(
            query, normalize_onsets(reference[:endpoint]), gap, max_step, unilateral
        )
    )


def _count_search(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    gap: float,
    max_step: int,
    unilateral: bool,
    prune: bool,
) -> PrefixMatch:
    endpoints = list(range(2, len(reference) + 1))
    if prune:
        endpoints.sort(key=lambda endpoint: (abs(endpoint - len(query)), endpoint))
    match = PrefixMatch()
    for endpoint in endpoints:
        margin = (
            1e-12 * max(1.0, abs(match.distance))
            if np.isfinite(match.distance)
            else 0.0
        )
        if prune and _count_bound(len(query), endpoint, gap) > match.distance + margin:
            continue
        score = _prefix_score(query, reference, endpoint, gap, max_step, unilateral)
        match = _update(match, score, endpoint)
    return match


def _roundoff_guard(
    query: NDArray[np.float64], reference: NDArray[np.float64]
) -> float:
    condition = 1 / np.min(np.diff(query)) + reference[-1] / np.min(np.diff(reference))
    return max(
        1e-10,
        64 * np.finfo(float).eps * (len(query) + len(reference)) * condition,
    )


def _tempo_search(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    gap: float,
    max_step: int,
) -> PrefixMatch:
    elapsed_reference = reference - reference[0]
    endpoints = sorted(
        range(2, len(reference) + 1),
        key=lambda endpoint: (abs(endpoint - len(query)), endpoint),
    )
    first = endpoints[0]
    score = _prefix_score(query, reference, first, gap, max_step, False)
    match = _update(PrefixMatch(), score, first)
    margin = _roundoff_guard(query, elapsed_reference)
    eligible = [
        endpoint
        for endpoint in endpoints
        if _count_bound(len(query), endpoint, gap) <= match.distance + margin
    ]
    extent = max(eligible)
    anchors = np.log2(elapsed_reference[first - 1]) + SHIFTS
    profiles = np.array(
        [
            _anchor_prefix_scores(
                query, elapsed_reference[:extent], gap, max_step, float(anchor)
            )
            for anchor in anchors
        ]
    )
    bounds = {}
    for endpoint in eligible:
        target = np.log2(elapsed_reference[endpoint - 1]) + SHIFTS
        timing = np.max(
            profiles[:, endpoint - 1, None] - abs(anchors[:, None] - target[None, :]),
            axis=0,
        )
        bounds[endpoint] = max(
            _count_bound(len(query), endpoint, gap),
            float(np.min(timing + 0.05 * abs(SHIFTS))),
        )
    ordered = sorted(
        (endpoint for endpoint in eligible if endpoint != first),
        key=lambda endpoint: (bounds[endpoint], endpoint),
    )
    for endpoint in ordered:
        if _count_bound(len(query), endpoint, gap) > match.distance + margin:
            continue
        if bounds[endpoint] > match.distance + margin:
            continue
        score = _prefix_score(query, reference, endpoint, gap, max_step, False)
        match = _update(match, score, endpoint)
    return match


def prefix_distance(
    query: Sequence[float],
    reference: Sequence[float],
    *,
    search: str = "tempo",
    family: str = "bilateral",
    gap: float = 1.0,
    max_step: int = 3,
) -> PrefixMatch:
    validate_parameters(gap, max_step)
    if family not in {"bilateral", "unilateral"}:
        raise ValueError("Family must be bilateral or unilateral.")
    if search not in {"none", "count", "tempo"}:
        raise ValueError("Search must be none, count or tempo.")
    if search == "tempo" and family == "unilateral":
        raise ValueError("Unilateral alignment supports none or count search.")
    query_onsets = canonical_onsets(query)
    reference_onsets = canonical_onsets(reference)
    if len(query_onsets) < 2 or len(reference_onsets) < 2:
        return PrefixMatch()
    normalized_query = normalize_onsets(query_onsets)
    normalize_onsets(reference_onsets)
    if search == "tempo":
        return _tempo_search(normalized_query, reference_onsets, gap, max_step)
    return _count_search(
        normalized_query,
        reference_onsets,
        gap,
        max_step,
        family == "unilateral",
        search == "count",
    )
