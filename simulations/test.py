import time

import numpy as np
from scipy.stats import norm

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

from randomization_test import randomization_based_confidence_interval
from simulation_helpers import (
    observed_counts_from_treated_counts,
    validate_alpha as _validate_alpha,
    validate_counts4 as _validate_counts4,
    validate_seed as _validate_seed,
)


def wald_ci_ht_counts(
    n_obs,
    alpha: float = 0.05,
    zcrit: float = 1.96,
):
    _validate_alpha(alpha)
    n_obs = _validate_counts4(n_obs, "n_obs")
    n = int(np.sum(n_obs))
    
    if n < 2:
        raise ValueError("Need n >= 2.")

    # the HT estimator
    tau_hat = 2.0 * (n_obs[0] - n_obs[2]) / n

    ss = (
        n_obs[0] * (2.0 - tau_hat) ** 2 +
        n_obs[1] * (0.0 - tau_hat) ** 2 +
        n_obs[2] * (-2.0 - tau_hat) ** 2 +
        n_obs[3] * (0.0 - tau_hat) ** 2
    )
    se = np.sqrt(ss / (n * (n - 1)))

    return (tau_hat - zcrit * se, tau_hat + zcrit * se)

def ci_test(
    v,
    alpha: float = 0.05,
    its: int = 10000,
    seed: int | None = 123,
):
    """
    Monte Carlo comparison of Wald and exact CIs for a 2x2 potential-outcome table.

    Args:
      v: [v11, v10, v01, v00] nonnegative integer counts.
      alpha: CI parameter.
      its: number of Monte Carlo draws.
      seed: RNG seed (None for nondeterministic entropy).
    """
    _validate_alpha(alpha)
    if its <= 0:
        raise ValueError(f"its must be positive, got {its}.")
    _validate_seed(seed)
    v = _validate_counts4(v, "v")
    n = int(np.sum(v))
    if n <= 0:
        raise ValueError("sum(v) must be positive.")

    tau = float(v[1] - v[2]) / float(n)
    print("outcome count: ", v, "alpha: ", alpha, "ATE: ", tau, "seed: ", seed)

    lo_wald = np.empty(its, dtype=float)
    hi_wald = np.empty(its, dtype=float)
    lo_exact = np.full(its, np.nan, dtype=float)
    hi_exact = np.full(its, np.nan, dtype=float)
    
    rng = np.random.default_rng(seed)
    zcrit = norm.ppf(1 - alpha/2)
    x_all = rng.binomial(v, 0.5, size=(its, 4))
    n_obs_all = np.empty((its, 4), dtype=int)

    start = time.time()
    for i in range(its):
        # if i % 1000 == 0:
        #     elapsed = time.time() - start
        #     print(f"trial {i}, elapsed {elapsed:.2f}s")

        x = x_all[i]
        n_obs = observed_counts_from_treated_counts(v, x)
        n_obs_all[i] = n_obs

        lo, hi = wald_ci_ht_counts(n_obs, alpha, zcrit=zcrit)
        lo_wald[i] = lo
        hi_wald[i] = hi
        
        tau_set = randomization_based_confidence_interval(n_obs, alpha)
        if not tau_set:
            # print(f"empty tau_set for n_obs={n_obs}")
            continue
        lo = min(tau_set)
        hi = max(tau_set)
        lo_exact[i] = lo
        hi_exact[i] = hi
    
    elapsed = time.time() - start
    print(f"trial {its}, elapsed time {elapsed:.2f}s")

    widths_wald = hi_wald - lo_wald
    widths_exact = hi_exact - lo_exact
    cover_wald = np.count_nonzero((lo_wald <= tau) & (tau <= hi_wald))
    cover_exact = np.count_nonzero((lo_exact <= tau) & (tau <= hi_exact))

    alpha_tag = int(round(alpha * 100))
    seed_tag = "none" if seed is None else f"{int(seed):010d}"
    np.savez(
        f"n_obs_{v[0]}_{v[1]}_{v[2]}_{v[3]}_alpha{alpha_tag:03d}_seed{seed_tag}.npz",
        n_obs_all,
    )
    np.savez(
        f"wald_exact_ci_bounds_{v[0]}_{v[1]}_{v[2]}_{v[3]}_alpha{alpha_tag:03d}_seed{seed_tag}.npz",
        lo_wald=lo_wald,
        hi_wald=hi_wald,
        lo_exact=lo_exact,
        hi_exact=hi_exact,
    )

    result = (
        np.median(widths_wald),
        np.nanmedian(widths_exact),
        100.0 * cover_wald / its,
        100.0 * cover_exact / its,
    )
    print(
        "Median width (Wald):", result[0],
        "Median width (Exact):", result[1],
        "Coverage % (Wald):", result[2],
        "Coverage % (Exact):", result[3],
    )
    
    return result

if __name__ == "__main__":
    # outcome count vector v = (v11, v10, v01, v00)
    v_list = [
        [10, 0, 0, 10],
        [25, 0, 0, 25],
        [50, 0, 0, 50],
        # [100, 0, 0, 100],
        # [500, 0, 0, 500],
        # [4, 0, 0, 46],
        # [8, 0, 0, 92],
        # [16, 0, 0, 184],
        # [80, 0, 0, 920],
    ]
    for v in v_list:
        result = ci_test(v, alpha=0.05, its=10000)
