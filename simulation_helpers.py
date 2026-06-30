from fractions import Fraction

import numpy as np


def validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def validate_seed(seed) -> None:
    if seed is None:
        return
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer or None, got {type(seed)}.")


def validate_counts4(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(arr, np.rint(arr))):
            raise ValueError(f"{name} entries must be integers.")
        arr = np.rint(arr).astype(int)
    else:
        arr = arr.astype(int, copy=False)
    if np.any(arr < 0):
        raise ValueError(f"{name} entries must be nonnegative.")
    return arr


def probability_fraction(
    p,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
) -> Fraction:
    if isinstance(p, Fraction):
        p_frac = p
    elif isinstance(p, str):
        p_frac = Fraction(p)
    else:
        p_float = float(p)
        p_frac = Fraction(p_float).limit_denominator(max_denominator)
        if abs(float(p_frac) - p_float) > p_tol:
            raise ValueError(
                "p is not close to a rational with denominator at most "
                f"{max_denominator}: got {p}."
            )
    if not (0 < p_frac < 1):
        raise ValueError(f"p must be in (0, 1), got {p}.")
    return p_frac


def probability_tag(p_frac: Fraction) -> str:
    return f"p{p_frac.numerator}_{p_frac.denominator}"


def observed_counts_from_treated_counts(v, x) -> list[int]:
    v11, v10, v01, v00 = [int(a) for a in v]
    x11, x10, x01, x00 = [int(a) for a in x]
    return [
        x11 + x10,
        x01 + x00,
        v11 + v01 - x11 - x01,
        v10 + v00 - x10 - x00,
    ]


def clip_ci_to_ate_range(lo: float, hi: float) -> tuple[float, float]:
    return (max(float(lo), -1.0), min(float(hi), 1.0))


def bernoulli_population_config(n: int, pattern: str) -> list[int]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if pattern == "balanced":
        if n % 2 != 0:
            raise ValueError(f"balanced pattern requires even n, got {n}.")
        return [n // 2, 0, 0, n // 2]
    if pattern == "asymmetric":
        n11 = int(np.ceil(0.08 * n))
        return [n11, 0, 0, n - n11]
    raise ValueError(f"unknown pattern: {pattern}")


def matched_pair_population_config(n: int, pattern: str) -> list[int]:
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"n must be a positive even integer, got {n}.")
    m = n // 2
    if pattern == "balanced":
        return [(m + 1) // 2, m // 2, 0, 0, 0, 0]
    if pattern == "asymmetric":
        m11 = int(np.ceil(0.92 * m))
        return [m11, m - m11, 0, 0, 0, 0]
    raise ValueError(f"unknown pattern: {pattern}")
