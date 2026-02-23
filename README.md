# randomization-ci

Python code for randomization-based confidence intervals for randomized experiments with binary outcomes under:
- Balanced Bernoulli design (`randomization_test.py`)
- Matched-pair designs (`matching.py`)

It also includes:
- FFT-based probability engine (`fft.py`)
- p-value maximization routine for balanced Bernoulli (`compute_pmax.py`)
- Rigdon-Hudgens comparison implementation (a variante for balanced Bernoulli) (`RigdonHudgens.py`)
- Monte Carlo simulation scripts (`test.py`, `matching.py`)

## Installation

```bash
pip install .
```

Optional matched-pair optimization dependency:

```bash
pip install .[mosek]
```

Development/test dependencies:

```bash
pip install .[dev]
```

## Quick Start

### Unmatched exact CI

```python
import numpy as np
from randomization_test import randomization_based_confidence_interval

n_obs = np.array([2, 6, 8, 0])  # [n11, n10, n01, n00]
alpha = 0.05
tau_set = randomization_based_confidence_interval(n_obs, alpha)
print(tau_set)
```

### Matched-pair exact CI

```python
from matching import permutation_based_confidence_interval_match

N_wz = {
    (-1, 0): 3, (-1, 1): 1,
    (0, 0): 2,  (0, 1): 2,
    (1, 0): 1,  (1, 1): 1,
}
alpha = 0.05
tau_set = permutation_based_confidence_interval_match(N_wz, alpha)
print(tau_set)
```

## Reproducibility

- Simulation entry points support RNG seeds.
- Output files from `test.py` include seed tags in filenames.

## Running Tests

```bash
pytest -q
```

## Release Checklist

1. Update `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Commit and tag:
   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```
4. Create a GitHub Release from the tag.

## License

MIT (see `LICENSE`).

