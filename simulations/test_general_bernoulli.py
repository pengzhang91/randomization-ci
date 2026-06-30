from fractions import Fraction
import time

import numpy as np
from scipy.stats import norm

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

from general_bernoulli import general_bernoulli_confidence_set
from simulation_helpers import (
    clip_ci_to_ate_range,
    observed_counts_from_treated_counts,
    probability_fraction as _probability_fraction,
    probability_tag as _p_tag,
    validate_alpha as _validate_alpha,
    validate_counts4 as _validate_counts4,
    validate_seed as _validate_seed,
)


def wald_ci_ht_counts_general_bernoulli(
    n_obs,
    p,
    alpha: float = 0.05,
    zcrit: float | None = None,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
):
    """
    Wald interval based on the Bernoulli(p) Horvitz-Thompson estimator.

    n_obs is ordered as (N11, N10, N01, N00), where the first index is Z and
    the second is the observed binary outcome Y.
    """
    _validate_alpha(alpha)
    if zcrit is None:
        zcrit = norm.ppf(1 - alpha / 2)
    n_obs = _validate_counts4(n_obs, "n_obs")
    p_frac = _probability_fraction(
        p,
        max_denominator=max_denominator,
        p_tol=p_tol,
    )
    p_float = float(p_frac)
    q_float = 1.0 - p_float
    n = int(np.sum(n_obs))
    if n < 2:
        raise ValueError("Need n >= 2.")

    tau_hat = n_obs[0] / (n * p_float) - n_obs[2] / (n * q_float)

    ss = (
        n_obs[0] * (1.0 / p_float - tau_hat) ** 2
        + n_obs[1] * (0.0 - tau_hat) ** 2
        + n_obs[2] * (-1.0 / q_float - tau_hat) ** 2
        + n_obs[3] * (0.0 - tau_hat) ** 2
    )
    se = np.sqrt(ss / (n * (n - 1)))

    return (tau_hat - zcrit * se, tau_hat + zcrit * se)

def ci_test_general_bernoulli(
    v,
    p,
    alpha: float = 0.05,
    its: int = 10000,
    seed: int | None = 123,
    save_outputs: bool = True,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
    clip_wald: bool = True,
):
    """
    Monte Carlo comparison of Wald and exact CIs under Bernoulli(p).

    Args:
      v: potential-outcome counts (v11, v10, v01, v00), where vij counts units
        with Y(1)=i and Y(0)=j.
      p: treatment probability. Prefer Fraction for exact rational p.
      alpha: CI parameter.
      its: number of Monte Carlo draws.
      seed: RNG seed (None for nondeterministic entropy).
      save_outputs: if True, write observed tables and CI bounds to .npz files.
      max_denominator, p_tol, eps_round: passed through to the exact routine.
      clip_wald: if True, intersect Wald intervals with the binary-outcome ATE
        range [-1, 1] before saving and summarizing.
    """
    _validate_alpha(alpha)
    if its <= 0:
        raise ValueError(f"its must be positive, got {its}.")
    _validate_seed(seed)
    v = _validate_counts4(v, "v")
    n = int(np.sum(v))
    if n <= 0:
        raise ValueError("sum(v) must be positive.")
    p_frac = _probability_fraction(
        p,
        max_denominator=max_denominator,
        p_tol=p_tol,
    )
    p_float = float(p_frac)

    tau = float(v[1] - v[2]) / float(n)
    print(
        "outcome count:",
        v,
        "p:",
        p_frac,
        "alpha:",
        alpha,
        "ATE:",
        tau,
        "seed:",
        seed,
    )

    lo_wald = np.empty(its, dtype=float)
    hi_wald = np.empty(its, dtype=float)
    lo_exact = np.full(its, np.nan, dtype=float)
    hi_exact = np.full(its, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    zcrit = norm.ppf(1 - alpha / 2)
    x_all = rng.binomial(v, p_float, size=(its, 4))
    n_obs_all = np.empty((its, 4), dtype=int)

    start = time.time()
    for i in range(its):
        n_obs = observed_counts_from_treated_counts(v, x_all[i])
        n_obs_all[i] = n_obs

        lo, hi = wald_ci_ht_counts_general_bernoulli(
            n_obs,
            p_frac,
            alpha,
            zcrit=zcrit,
            max_denominator=max_denominator,
            p_tol=p_tol,
        )
        if clip_wald:
            lo, hi = clip_ci_to_ate_range(lo, hi)
        lo_wald[i] = lo
        hi_wald[i] = hi

        tau_set = general_bernoulli_confidence_set(
            n_obs,
            p_frac,
            alpha,
            max_denominator=max_denominator,
            p_tol=p_tol,
            eps_round=eps_round,
        )
        if not tau_set:
            continue
        lo_exact[i] = min(tau_set)
        hi_exact[i] = max(tau_set)

    elapsed = time.time() - start
    print(f"trial {its}, elapsed time {elapsed:.2f}s")

    widths_wald = hi_wald - lo_wald
    widths_exact = hi_exact - lo_exact
    cover_wald = np.count_nonzero((lo_wald <= tau) & (tau <= hi_wald))
    cover_exact = np.count_nonzero((lo_exact <= tau) & (tau <= hi_exact))

    if save_outputs:
        alpha_tag = int(round(alpha * 100))
        seed_tag = "none" if seed is None else f"{int(seed):010d}"
        p_tag = _p_tag(p_frac)
        wald_tag = "waldclip" if clip_wald else "waldraw"
        base = (
            f"{v[0]}_{v[1]}_{v[2]}_{v[3]}_{p_tag}_alpha{alpha_tag:03d}_"
            f"seed{seed_tag}_{wald_tag}"
        )
        np.savez(f"gb_n_obs_{base}.npz", n_obs_all=n_obs_all)
        np.savez(
            f"gb_wald_exact_ci_bounds_{base}.npz",
            lo_wald=lo_wald,
            hi_wald=hi_wald,
            lo_exact=lo_exact,
            hi_exact=hi_exact,
            clip_wald=clip_wald,
        )

    result = (
        np.median(widths_wald),
        np.nanmedian(widths_exact),
        100.0 * cover_wald / its,
        100.0 * cover_exact / its,
    )
    print(
        "Median width (Wald):",
        result[0],
        "Median width (Exact):",
        result[1],
        "Coverage % (Wald):",
        result[2],
        "Coverage % (Exact):",
        result[3],
    )

    return result


if __name__ == "__main__":
    # Potential-outcome count vector v = (v11, v10, v01, v00).
    v_list = [
        [10, 0, 0, 10],
        [25, 0, 0, 25],
        [4, 0, 0, 46],
    ]
    for v in v_list:
        ci_test_general_bernoulli(v, p=Fraction(2, 5), alpha=0.05, its=1000)
