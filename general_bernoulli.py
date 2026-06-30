from fractions import Fraction
import math

import numpy as np

from fft import compute_p_v_p_fft


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def _validate_n_obs(n_obs) -> np.ndarray:
    arr = np.asarray(n_obs)
    if arr.shape != (4,):
        raise ValueError(f"n_obs must have shape (4,), got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(arr, np.rint(arr))):
            raise ValueError("n_obs entries must be integers.")
        arr = np.rint(arr).astype(int)
    else:
        arr = arr.astype(int, copy=False)
    if np.any(arr < 0):
        raise ValueError("n_obs entries must be nonnegative.")
    if int(np.sum(arr)) <= 0:
        raise ValueError("Total sample size must be positive.")
    return arr


def _probability_fraction(p, max_denominator: int, p_tol: float) -> Fraction:
    if not isinstance(max_denominator, (int, np.integer)) or int(max_denominator) <= 0:
        raise ValueError(
            f"max_denominator must be a positive integer, got {max_denominator}."
        )
    if p_tol < 0:
        raise ValueError(f"p_tol must be nonnegative, got {p_tol}.")

    if isinstance(p, Fraction):
        p_frac = p
    elif isinstance(p, str):
        p_frac = Fraction(p)
    else:
        p_float = float(p)
        p_frac = Fraction(p_float).limit_denominator(int(max_denominator))
        if abs(float(p_frac) - p_float) > p_tol:
            raise ValueError(
                "p is not close to a rational with denominator at most "
                f"{max_denominator}: got {p}."
            )

    if not (0 < p_frac < 1):
        raise ValueError(f"p must be in (0, 1), got {p}.")
    return p_frac


def observed_ht_statistic_general_bernoulli(n_obs, p) -> float:
    """
    Return the observed Horvitz-Thompson statistic under Bernoulli(p).

    n_obs is ordered as (N11, N10, N01, N00), where the first index is Z and
    the second index is the observed binary outcome.
    """
    n_obs = _validate_n_obs(n_obs)
    p_frac = _probability_fraction(p, max_denominator=10_000, p_tol=1e-12)
    p_float = float(p_frac)
    n = int(np.sum(n_obs))
    return float(n_obs[0] / (n * p_float) - n_obs[2] / (n * (1.0 - p_float)))


def candidate_tau_ints_eq14(
    n_obs,
    p,
    alpha: float,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
) -> np.ndarray:
    """
    Return integer k values for tau_0 = k/n in the Eq. (14) candidate set.
    """
    _validate_alpha(alpha)
    n_obs = _validate_n_obs(n_obs)
    p_frac = _probability_fraction(p, max_denominator, p_tol)
    p_float = float(p_frac)
    q_float = 1.0 - p_float
    n = int(np.sum(n_obs))
    t_obs = float(n_obs[0] / (n * p_float) - n_obs[2] / (n * q_float))

    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * n * p_float**2 * q_float**2))
    lo = max(-n, math.ceil(n * (t_obs - radius) - 1e-12))
    hi = min(n, math.floor(n * (t_obs + radius) + 1e-12))
    if lo > hi:
        return np.array([], dtype=int)
    return np.arange(lo, hi + 1, dtype=int)



def compute_p_max_general_bernoulli(
    n: int,
    n_obs,
    p,
    tau_int: int,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
    alpha_stop: float | None = None,
) -> float:
    """
    Compute max_v p^{Ber(p)}(v, Delta) for one tau_0 = tau_int / n.

    If alpha_stop is provided, return as soon as a p-value >= alpha_stop is
    found. This is enough for Algorithm 2, which only tests p_max >= alpha.
    """
    if alpha_stop is not None:
        _validate_alpha(alpha_stop)
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    n = int(n)
    n_obs = _validate_n_obs(n_obs)
    if int(np.sum(n_obs)) != n:
        raise ValueError(
            f"sum(n_obs) must equal n, got sum(n_obs)={int(np.sum(n_obs))}, n={n}."
        )
    if not isinstance(tau_int, (int, np.integer)):
        raise ValueError(f"tau_int must be an integer, got {tau_int}.")
    tau_int = int(tau_int)
    if abs(tau_int) > n:
        return 0.0

    p_frac = _probability_fraction(p, max_denominator, p_tol)
    p_float = float(p_frac)
    q_float = 1.0 - p_float
    t_obs = float(n_obs[0] / (n * p_float) - n_obs[2] / (n * q_float))
    tau_0 = tau_int / n
    Delta = n * abs(t_obs - tau_0)

    p_max = 0.0

    n_tau_abs = abs(tau_int)
    N1_star = n_obs[0] + n_obs[2]
    U = min(2*n - tau_int - 2*n_obs[1], 2*n + tau_int - 2*n_obs[3])
    L = max(2*n_obs[0] - tau_int, 2*n_obs[2] + tau_int)
    Ub = min(
        2 * (n_obs[0] + n_obs[3]) - tau_int,
        2 * (n_obs[1] + n_obs[2]) + tau_int,
    )

    def iter_candidates():
        if alpha_stop is None:
            v11_range = range(N1_star + 1)
        else:
            v11_range = range(N1_star, -1, -1)

        for v11 in v11_range:
            b_lower = max(n_tau_abs, N1_star - v11, L - 2*v11)
            b_upper = min(Ub, n-v11, U-2*v11)
            if alpha_stop is None:
                b_start = b_lower
                if (b_start - n_tau_abs) % 2 != 0:
                    b_start += 1
                b_range = range(b_start, b_upper + 1, 2)
            else:
                b_start = b_upper
                if (b_start - n_tau_abs) % 2 != 0:
                    b_start -= 1
                b_range = range(b_start, b_lower - 1, -2)

            for b in b_range:
                v10 = (b + tau_int) // 2
                v01 = (b - tau_int) // 2
                yield v11, v10, v01

    for v11, v10, v01 in iter_candidates():

        p_value = compute_p_v_p_fft(
            v11,
            v10,
            v01,
            p_frac,
            Delta,
            max_denominator=max_denominator,
            p_tol=p_tol,
            eps_round=eps_round,
        )
        if p_value > p_max:
            p_max = p_value
            if alpha_stop is not None and p_max >= alpha_stop:
                return float(p_max)

    return float(p_max)


def general_bernoulli_confidence_set(
    n_obs,
    p,
    alpha: float,
    verbose: bool = False,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
):
    """
    Algorithm 2 from page 20: exact confidence set for general Bernoulli(p).

    Args:
      n_obs: observed counts (N11, N10, N01, N00), first index Z, second
        observed binary outcome.
      p: treatment probability in (0, 1). Prefer Fraction for exact rational p.
      alpha: significance level.
      verbose: if True, print candidate and p-value call counts.
      max_denominator: denominator bound used when converting float p.
      p_tol: tolerance for accepting rational approximation to float p.
      eps_round: FFT rounding tolerance.

    Returns:
      Sorted list of accepted ATE candidates tau_0.
    """
    _validate_alpha(alpha)
    n_obs = _validate_n_obs(n_obs)
    p_frac = _probability_fraction(p, max_denominator, p_tol)
    n = int(np.sum(n_obs))

    tau_ints = candidate_tau_ints_eq14(
        n_obs,
        p_frac,
        alpha,
        max_denominator=max_denominator,
        p_tol=p_tol,
    )
    accepted = []
    p_value_calls = 0

    for tau_int in tau_ints:
        p_max = compute_p_max_general_bernoulli(
            n,
            n_obs,
            p_frac,
            int(tau_int),
            max_denominator=max_denominator,
            p_tol=p_tol,
            eps_round=eps_round,
            alpha_stop=alpha,
        )
        p_value_calls += 1
        if p_max >= alpha:
            accepted.append(int(tau_int) / n)

    if verbose:
        print("Algorithm 2 candidate taus:", len(tau_ints))
        print("Algorithm 2 p_max calls:", p_value_calls)

    return accepted


def randomization_based_confidence_interval_general_bernoulli(
    n_obs,
    p,
    alpha: float,
    verbose: bool = False,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
):
    return general_bernoulli_confidence_set(
        n_obs,
        p,
        alpha,
        verbose=verbose,
        max_denominator=max_denominator,
        p_tol=p_tol,
        eps_round=eps_round,
    )


algorithm2_general_bernoulli_confidence_set = general_bernoulli_confidence_set
