import argparse
import csv
import time

import numpy as np

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

import matching
from simulation_helpers import matched_pair_population_config as population_config_pairs


DEFAULT_N_LIST = [20, 50, 100, 200, 500, 1000, 2000, 4000, 10000]


def _validate_v6(v) -> np.ndarray:
    arr = np.asarray(v)
    if arr.shape != (6,):
        raise ValueError(f"v must have shape (6,), got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(arr, np.rint(arr))):
            raise ValueError("v entries must be integers.")
        arr = np.rint(arr).astype(int)
    else:
        arr = arr.astype(int, copy=False)
    if np.any(arr < 0):
        raise ValueError("v entries must be nonnegative.")
    if int(np.sum(arr)) <= 0:
        raise ValueError("sum(v) must be positive.")
    return arr


def observed_tables_from_pair_population(v, its: int, seed: int | None) -> list[dict]:
    v = _validate_v6(v)
    rng = np.random.default_rng(seed)
    x_all = rng.binomial(v, 0.5, size=(its, 6))

    out = []
    for x in x_all:
        out.append(
            {
                (-1, 0): int(v[2] - x[2] + v[4] - x[4] + v[5] - x[5]),
                (-1, 1): int(x[5]),
                (0, 0): int(v[1] - x[1] + v[3] - x[3]),
                (0, 1): int(x[3] + x[4]),
                (1, 0): int(v[0] - x[0]),
                (1, 1): int(x[0] + x[1] + x[2]),
            }
        )
    return out


def benchmark_expected_runtime(v, alpha: float, its: int, seed: int | None) -> dict:
    N_wz_all = observed_tables_from_pair_population(v, its=its, seed=seed)
    times = np.empty(its, dtype=float)
    sizes = np.empty(its, dtype=int)
    p_v_calls = np.empty(its, dtype=int)

    original_compute_p_v_fft = matching.compute_p_v_fft
    call_count = 0

    def counted_compute_p_v_fft(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_compute_p_v_fft(*args, **kwargs)

    matching.compute_p_v_fft = counted_compute_p_v_fft
    try:
        for i, N_wz in enumerate(N_wz_all):
            before = call_count
            start = time.perf_counter()
            tau_set = matching.permutation_based_confidence_interval_match(N_wz, alpha)
            times[i] = time.perf_counter() - start
            sizes[i] = len(tau_set)
            p_v_calls[i] = call_count - before
    finally:
        matching.compute_p_v_fft = original_compute_p_v_fft

    unique_tables = {
        tuple(N_wz[key] for key in [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)])
        for N_wz in N_wz_all
    }
    v = _validate_v6(v)
    return {
        "v": v.tolist(),
        "m": int(np.sum(v)),
        "n": int(2 * np.sum(v)),
        "its": its,
        "unique_N_wz": int(len(unique_tables)),
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
        description="Benchmark expected runtime of matched-pair exact CI construction."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--its", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--n-list",
        nargs="+",
        type=int,
        default=DEFAULT_N_LIST,
        help="Even unit sample sizes n to benchmark; matched pairs m=n/2.",
    )
    parser.add_argument(
        "--pattern",
        choices=["balanced", "asymmetric"],
        default="balanced",
        help=(
            "Pair-level population pattern. Balanced gives "
            "[ceil(m/2), floor(m/2), 0, 0, 0, 0]. Asymmetric gives "
            "[ceil(0.92*m), m-ceil(0.92*m), 0, 0, 0, 0]."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path. Rows are appended as each n finishes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fieldnames = [
        "pattern",
        "v",
        "m",
        "n",
        "its",
        "unique_N_wz",
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
        v = population_config_pairs(n, args.pattern)
        result = benchmark_expected_runtime(
            v, alpha=args.alpha, its=args.its, seed=args.seed
        )
        row = {"pattern": args.pattern, **result}
        print(
            "{pattern},{v},{m},{n},{its},{unique_N_wz},{mean_s:.6f},{median_s:.6f},"
            "{p10_s:.6f},"
            "{p90_s:.6f},{p95_s:.6f},{max_s:.6f},{total_s:.6f},{mean_ci_size:.2f},"
            "{total_p_v_calls},{mean_p_v_calls:.2f},{median_p_v_calls:.2f},"
            "{p10_p_v_calls:.2f},{p90_p_v_calls:.2f}".format(
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
