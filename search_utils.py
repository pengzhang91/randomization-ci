import numpy as np


def binary_search_max_true(predicate, lo: int, hi: int) -> int:
    """
    Return the maximum index k in [lo, hi] (inclusive) such that predicate(k) is True.
    If none are True, return -1.
    Assumes indices are integers and predicate is monotone (True then False after).
    """
    res = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            res = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return res


def binary_search_min_true(predicate, lo: int, hi: int) -> int:
    """
    Return the minimum index k in [lo, hi] (inclusive) such that predicate(k) is True.
    If none are True, return -1.
    Assumes predicate is monotone (False then True after).
    """
    res = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            res = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return res


def confidence_interval_from_candidates(taus, t_obs: float, alpha: float, p_value_fn):
    """
    Return the accepted candidate values from a monotone randomization test.

    Args:
      taus: sorted 1D array-like of candidate values.
      t_obs: observed test statistic value.
      alpha: significance threshold.
      p_value_fn: callable mapping tau -> p_max(tau).
    """
    taus = np.asarray(taus, dtype=float)
    if taus.ndim != 1:
        raise ValueError("taus must be a one-dimensional array.")
    if taus.size == 0:
        return []

    k = taus.size

    if t_obs < taus[0]:
        k_max_idx = binary_search_max_true(lambda idx: p_value_fn(taus[idx]) >= alpha, 0, k - 1)
        return [] if k_max_idx == -1 else taus[:k_max_idx + 1].tolist()

    if t_obs > taus[-1]:
        k_min_idx = binary_search_min_true(lambda idx: p_value_fn(taus[idx]) >= alpha, 0, k - 1)
        return [] if k_min_idx == -1 else taus[k_min_idx:].tolist()

    k_minus = int(np.max(np.where(taus <= t_obs)[0]))
    k_plus = int(np.min(np.where(taus >= t_obs)[0]))

    k_min_idx = binary_search_min_true(lambda idx: p_value_fn(taus[idx]) >= alpha, 0, k_minus)
    if k_min_idx == -1:
        k_min_idx = k_plus

    k_max_idx = binary_search_max_true(lambda idx: p_value_fn(taus[idx]) >= alpha, k_plus, k - 1)
    if k_max_idx == -1:
        k_max_idx = k_minus

    if k_min_idx > k_max_idx:
        return []
    return taus[k_min_idx:k_max_idx + 1].tolist()
