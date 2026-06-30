import argparse
import csv
from fractions import Fraction
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.stats import norm

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

import general_bernoulli
from simulation_helpers import (
    bernoulli_population_config as population_config,
    clip_ci_to_ate_range,
    observed_counts_from_treated_counts,
)
from test_general_bernoulli import (
    wald_ci_ht_counts_general_bernoulli,
)


DEFAULT_N_LIST = [20, 50, 100]


def parse_probability(value: str) -> Fraction:
    try:
        p = Fraction(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid probability {value!r}") from exc
    if not (0 < p < 1):
        raise argparse.ArgumentTypeError(f"p must be in (0, 1), got {value!r}")
    return p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Table-4-style general Bernoulli CI coverage/width tests."
    )
    parser.add_argument("--pattern", choices=["balanced", "asymmetric"], required=True)
    parser.add_argument(
        "--p",
        type=parse_probability,
        default=Fraction(2, 5),
        help="Bernoulli treatment probability, e.g. 2/5 or 0.4.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--its", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-list", nargs="+", type=int, default=DEFAULT_N_LIST)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for Monte Carlo replicates. "
            "Use 1 for serial execution, or 0 for all available CPUs."
        ),
    )
    parser.add_argument(
        "--no-clip-wald",
        action="store_true",
        help="Do not clip Wald intervals to the ATE range [-1, 1].",
    )
    return parser.parse_args()


def _one_ci_replicate(args):
    v, x, p, alpha, zcrit, clip_wald = args
    n = int(np.sum(v))
    tau = float(v[1] - v[2]) / float(n)
    n_obs = observed_counts_from_treated_counts(v, x)

    lo_wald, hi_wald = wald_ci_ht_counts_general_bernoulli(
        n_obs,
        p,
        alpha,
        zcrit=zcrit,
    )
    if clip_wald:
        lo_wald, hi_wald = clip_ci_to_ate_range(lo_wald, hi_wald)

    call_count = 0
    original_compute_p_v_p_fft = general_bernoulli.compute_p_v_p_fft

    def counted_compute_p_v_p_fft(*call_args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_compute_p_v_p_fft(*call_args, **kwargs)

    general_bernoulli.compute_p_v_p_fft = counted_compute_p_v_p_fft
    try:
        start = time.perf_counter()
        tau_set = general_bernoulli.general_bernoulli_confidence_set(
            n_obs,
            p,
            alpha,
        )
        exact_time = time.perf_counter() - start
    finally:
        general_bernoulli.compute_p_v_p_fft = original_compute_p_v_p_fft

    lo_exact = np.nan
    hi_exact = np.nan
    if tau_set:
        lo_exact = min(tau_set)
        hi_exact = max(tau_set)

    return lo_wald, hi_wald, lo_exact, hi_exact, exact_time, call_count, tau


def ci_test_general_bernoulli_with_metrics(
    v,
    p,
    alpha: float,
    its: int,
    seed: int | None,
    clip_wald: bool,
    workers: int,
):
    v = np.asarray(v, dtype=int)
    n = int(np.sum(v))
    p_float = float(p)
    tau = float(v[1] - v[2]) / float(n)

    print(
        "outcome count:",
        v,
        "p:",
        p,
        "alpha:",
        alpha,
        "ATE:",
        tau,
        "seed:",
        seed,
        flush=True,
    )

    lo_wald = np.empty(its, dtype=float)
    hi_wald = np.empty(its, dtype=float)
    lo_exact = np.full(its, np.nan, dtype=float)
    hi_exact = np.full(its, np.nan, dtype=float)
    exact_times = np.empty(its, dtype=float)
    p_v_calls = np.empty(its, dtype=int)

    rng = np.random.default_rng(seed)
    zcrit = norm.ppf(1 - alpha / 2)
    x_all = rng.binomial(v, p_float, size=(its, 4))

    start_all = time.perf_counter()
    tasks = [(v, x_all[i], p, alpha, zcrit, clip_wald) for i in range(its)]
    if workers == 1:
        results = [_one_ci_replicate(task) for task in tasks]
    else:
        if workers <= 0:
            workers = os.cpu_count() or 1
        chunksize = max(1, its // (4 * workers))
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_one_ci_replicate, tasks, chunksize=chunksize)

    for i, (lo_w, hi_w, lo_e, hi_e, exact_time, calls, _) in enumerate(results):
        lo_wald[i] = lo_w
        hi_wald[i] = hi_w
        lo_exact[i] = lo_e
        hi_exact[i] = hi_e
        exact_times[i] = exact_time
        p_v_calls[i] = calls

    elapsed = time.perf_counter() - start_all
    print(f"trial {its}, elapsed time {elapsed:.2f}s", flush=True)

    widths_wald = hi_wald - lo_wald
    widths_exact = hi_exact - lo_exact
    cover_wald = np.count_nonzero((lo_wald <= tau) & (tau <= hi_wald))
    cover_exact = np.count_nonzero((lo_exact <= tau) & (tau <= hi_exact))

    result = {
        "median_width_wald": float(np.median(widths_wald)),
        "p10_width_wald": float(np.quantile(widths_wald, 0.10)),
        "p90_width_wald": float(np.quantile(widths_wald, 0.90)),
        "median_width_exact": float(np.nanmedian(widths_exact)),
        "p10_width_exact": float(np.nanquantile(widths_exact, 0.10)),
        "p90_width_exact": float(np.nanquantile(widths_exact, 0.90)),
        "coverage_wald_pct": float(100.0 * cover_wald / its),
        "coverage_exact_pct": float(100.0 * cover_exact / its),
        "median_runtime_s": float(np.median(exact_times)),
        "mean_runtime_s": float(np.mean(exact_times)),
        "p10_runtime_s": float(np.quantile(exact_times, 0.10)),
        "p90_runtime_s": float(np.quantile(exact_times, 0.90)),
        "total_runtime_s": float(np.sum(exact_times)),
        "median_p_v_calls": float(np.median(p_v_calls)),
        "mean_p_v_calls": float(np.mean(p_v_calls)),
        "p10_p_v_calls": float(np.quantile(p_v_calls, 0.10)),
        "p90_p_v_calls": float(np.quantile(p_v_calls, 0.90)),
        "total_p_v_calls": int(np.sum(p_v_calls)),
    }
    print(
        "Median width (Wald):",
        result["median_width_wald"],
        "Median width (Exact):",
        result["median_width_exact"],
        "Coverage % (Wald):",
        result["coverage_wald_pct"],
        "Coverage % (Exact):",
        result["coverage_exact_pct"],
        "Median runtime s:",
        result["median_runtime_s"],
        "Median p_v calls:",
        result["median_p_v_calls"],
        flush=True,
    )
    return result


def main():
    args = parse_args()
    fieldnames = [
        "pattern",
        "p",
        "v",
        "n",
        "alpha",
        "its",
        "seed",
        "clip_wald",
        "median_width_wald",
        "p10_width_wald",
        "p90_width_wald",
        "median_width_exact",
        "p10_width_exact",
        "p90_width_exact",
        "coverage_wald_pct",
        "coverage_exact_pct",
        "median_runtime_s",
        "mean_runtime_s",
        "p10_runtime_s",
        "p90_runtime_s",
        "total_runtime_s",
        "median_p_v_calls",
        "mean_p_v_calls",
        "p10_p_v_calls",
        "p90_p_v_calls",
        "total_p_v_calls",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for n in args.n_list:
            v = population_config(n, args.pattern)
            result = ci_test_general_bernoulli_with_metrics(
                v,
                p=args.p,
                alpha=args.alpha,
                its=args.its,
                seed=args.seed,
                clip_wald=not args.no_clip_wald,
                workers=args.workers,
            )
            row = {
                "pattern": args.pattern,
                "p": str(args.p),
                "v": v,
                "n": n,
                "alpha": args.alpha,
                "its": args.its,
                "seed": args.seed,
                "clip_wald": not args.no_clip_wald,
                **result,
            }
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
