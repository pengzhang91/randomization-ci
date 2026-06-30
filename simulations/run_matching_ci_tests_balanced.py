import argparse
import csv

try:
    from ._repo_path import add_repo_root
except ImportError:
    from _repo_path import add_repo_root

add_repo_root(__file__)

from matching import ci_test_match
from simulation_helpers import matched_pair_population_config


DEFAULT_N_LIST = [50, 100, 200, 1000]


def balanced_v_for_n(n: int) -> list[int]:
    return matched_pair_population_config(n, "balanced")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched-pair CI coverage/width tests for balanced configurations."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--its", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-list", nargs="+", type=int, default=DEFAULT_N_LIST)
    parser.add_argument(
        "--output",
        default="matching_ci_test_balanced_its10000.csv",
        help="CSV output path. Rows are written as each n finishes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fieldnames = [
        "pattern",
        "v",
        "m",
        "n",
        "alpha",
        "its",
        "seed",
        "median_width_wald",
        "median_width_exact",
        "coverage_wald_pct",
        "coverage_exact_pct",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for n in args.n_list:
            v = balanced_v_for_n(n)
            result = ci_test_match(v, alpha=args.alpha, its=args.its, seed=args.seed)
            row = {
                "pattern": "balanced",
                "v": v,
                "m": n // 2,
                "n": n,
                "alpha": args.alpha,
                "its": args.its,
                "seed": args.seed,
                "median_width_wald": result[0],
                "median_width_exact": result[1],
                "coverage_wald_pct": result[2],
                "coverage_exact_pct": result[3],
            }
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
