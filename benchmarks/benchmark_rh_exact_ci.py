import argparse
import csv
import time

import numpy as np

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

import RigdonHudgens
from benchmark_exact_ci import observed_tables_from_population, population_config


DEFAULT_N_LIST = [20, 50, 100]


def benchmark_expected_runtime(v, alpha: float, its: int, seed: int | None) -> dict:
    n_obs_all = observed_tables_from_population(v, its=its, seed=seed)
    times = np.empty(its, dtype=float)
    sizes = np.empty(its, dtype=int)
    p_v_calls = np.empty(its, dtype=int)

    original_compute_p_v_fft = RigdonHudgens.compute_p_v_fft
    call_count = 0

    def counted_compute_p_v_fft(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_compute_p_v_fft(*args, **kwargs)

    RigdonHudgens.compute_p_v_fft = counted_compute_p_v_fft
    try:
        for i, n_obs in enumerate(n_obs_all):
            before = call_count
            start = time.perf_counter()
            tau_set = RigdonHudgens.Rigdon_Hudgens(n_obs, alpha)
            times[i] = time.perf_counter() - start
            sizes[i] = len(tau_set)
            p_v_calls[i] = call_count - before
    finally:
        RigdonHudgens.compute_p_v_fft = original_compute_p_v_fft

    v = np.asarray(v, dtype=int)
    return {
        "v": v.tolist(),
        "n": int(np.sum(v)),
        "its": its,
        "unique_n_obs": int(len({tuple(row) for row in n_obs_all})),
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "p10_s": float(np.quantile(times, 0.10)),
        "p90_s": float(np.quantile(times, 0.90)),
        "p95_s": float(np.quantile(times, 0.95)),
        "max_s": float(np.max(times)),
        "total_s": float(np.sum(times)),
        "mean_ci_size": float(np.mean(sizes)),
        "total_p_v_calls": int(np.sum(p_v_calls)),
        "mean_p_v_calls": float(np.mean(p_v_calls)),
        "median_p_v_calls": float(np.median(p_v_calls)),
        "p10_p_v_calls": float(np.quantile(p_v_calls, 0.10)),
        "p90_p_v_calls": float(np.quantile(p_v_calls, 0.90)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Rigdon-Hudgens enumeration runtime for Bernoulli exact CIs."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--its", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--n-list",
        nargs="+",
        type=int,
        default=DEFAULT_N_LIST,
        help="Sample sizes n to benchmark.",
    )
    parser.add_argument(
        "--pattern",
        choices=["balanced", "asymmetric"],
        default="balanced",
        help=(
            "Population pattern: balanced gives [n/2, 0, 0, n/2]; "
            "asymmetric gives [ceil(0.08*n), 0, 0, n-ceil(0.08*n)]."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path. Rows are written as each n finishes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fieldnames = [
        "pattern",
        "v",
        "n",
        "its",
        "unique_n_obs",
        "mean_s",
        "median_s",
        "p10_s",
        "p90_s",
        "p95_s",
        "max_s",
        "total_s",
        "mean_ci_size",
        "total_p_v_calls",
        "mean_p_v_calls",
        "median_p_v_calls",
        "p10_p_v_calls",
        "p90_p_v_calls",
    ]
    print(",".join(fieldnames))
    csv_file = open(args.output, "w", newline="") if args.output else None
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames) if csv_file else None
    if writer is not None:
        writer.writeheader()
        csv_file.flush()

    for n in args.n_list:
        v = population_config(n, args.pattern)
        result = benchmark_expected_runtime(
            v, alpha=args.alpha, its=args.its, seed=args.seed
        )
        row = {"pattern": args.pattern, **result}
        print(
            "{pattern},{v},{n},{its},{unique_n_obs},{mean_s:.6f},{median_s:.6f},"
            "{p10_s:.6f},{p90_s:.6f},{p95_s:.6f},{max_s:.6f},{total_s:.6f},"
            "{mean_ci_size:.2f},{total_p_v_calls},{mean_p_v_calls:.2f},"
            "{median_p_v_calls:.2f},{p10_p_v_calls:.2f},{p90_p_v_calls:.2f}".format(
                **row
            ),
            flush=True,
        )
        if writer is not None:
            writer.writerow(row)
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
