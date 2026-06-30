# randomization-ci simulation code

Python code for paper ''Fast exact confidence sets for randomized experiments
with binary outcomes'' by Peng Zhang.

## Contents

- `randomization_test.py`, `compute_pmax.py`, `RigdonHudgens.py`: symmetric Bernoulli (`p=1/2`) exact CI routines and comparison code.
- `matching.py`: matched-pair exact CI and Wald interval routines.
- `general_bernoulli.py`: exact confidence sets for Bernoulli(`p`) assignment.
- `simulations/`: Monte Carlo simulation entry points and simulation helpers.
- `benchmarks/`: benchmark entry points and benchmark shell scripts.
- `tests/`: quick automated tests for CI.

Generated outputs such as CSV, NPZ, PNG, PDF, and log files are intentionally ignored by `.gitignore`.

## Installation

```bash
pip install .
```

For development tests:

```bash
pip install .[dev]
pytest -q
```

The matched-pair simulation uses MOSEK for solving integer linear programming:

```bash
pip install .[mosek]
```

## Symmetric Bernoulli Simulation

The symmetric Bernoulli simulation compares Wald intervals with exact intervals under treatment probability `p=1/2`.

```bash
python simulations/test.py
```

The outcome settings are listed in `v_list` at the bottom of `simulations/test.py`, where each vector is ordered as `(v11, v10, v01, v00)`. The script reports median interval width and coverage for Wald and exact intervals, and writes the simulated observed tables and CI bounds to ignored NPZ files.


## Matched-Pair Simulation

Example: run matched-pair coverage and width simulations for the balanced setting.

```bash
python simulations/run_matching_ci_tests.py \
  --pattern balanced \
  --alpha 0.05 \
  --its 1000 \
  --seed 123 \
  --n-list 50 100 200 1000 \
  --output matching_ci_test_balanced_its1000.csv
```

For the asymmetric setting, change `--pattern` and the output path:

```bash
python simulations/run_matching_ci_tests.py \
  --pattern asymmetric \
  --alpha 0.05 \
  --its 1000 \
  --seed 123 \
  --n-list 50 100 200 1000 \
  --output matching_ci_test_asymmetric_its1000.csv
```

The output CSV includes Wald and exact median widths and coverage percentages. The exact matched-pair routine requires MOSEK.

## General Bernoulli Simulation

Example: reproduce table-style simulations for `p=1/4`, sample sizes `50, 100, 200`, and 5000 Monte Carlo replicates.

```bash
python simulations/run_general_bernoulli_ci_tests.py \
  --pattern balanced \
  --p 1/4 \
  --alpha 0.05 \
  --its 5000 \
  --seed 123 \
  --n-list 50 100 200 \
  --workers 4 \
  --output general_bernoulli_ci_balanced_p1_4_its5000_workers4.csv
```

```bash
python simulations/run_general_bernoulli_ci_tests.py \
  --pattern asymmetric \
  --p 1/4 \
  --alpha 0.05 \
  --its 5000 \
  --seed 123 \
  --n-list 50 100 200 \
  --workers 4 \
  --output general_bernoulli_ci_asymmetric_p1_4_its5000_workers4.csv
```

Use `--workers 1` for serial execution, a positive integer for that many worker processes, or `--workers 0` to use all available CPUs. When running multiple jobs at once, set `--workers` conservatively to avoid oversubscribing the machine.

The output CSV includes Wald and exact CI width summaries, coverage, exact-CI runtime summaries, and p-value call counts.

## Quick API Examples


### Symmetric Bernoulli exact CI

```python
from randomization_test import randomization_based_confidence_interval

n_obs = [2, 6, 8, 0]
tau_set = randomization_based_confidence_interval(n_obs, alpha=0.05)
print(tau_set)
```

### Matched-pair exact CI

This exact matched-pair example requires MOSEK and a valid MOSEK license.

```python
from matching import permutation_based_confidence_interval_match

N_wz = {
    (-1, 0): 3, (-1, 1): 1,
    (0, 0): 2,  (0, 1): 2,
    (1, 0): 1,  (1, 1): 1,
}
tau_set = permutation_based_confidence_interval_match(N_wz, alpha=0.05)
print(tau_set)
```

### General Bernoulli exact CI

```python
from fractions import Fraction
from general_bernoulli import general_bernoulli_confidence_set

n_obs = [2, 6, 8, 0]  # (N11, N01, N10, N00)
tau_set = general_bernoulli_confidence_set(n_obs, p=Fraction(1, 4), alpha=0.05)
print(tau_set)
```

## License

MIT. See `LICENSE`.
