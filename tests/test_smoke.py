import numpy as np

from fft import compute_p_v_fft
from randomization_test import randomization_based_confidence_interval
from RigdonHudgens import Rigdon_Hudgens


def test_fft_probability_bounds():
    p = compute_p_v_fft(2, 3, 1.0)
    assert 0.0 <= p <= 1.0


def test_randomization_ci_returns_grid_values():
    n_obs = np.array([2, 6, 8, 0])
    alpha = 0.05
    tau_set = randomization_based_confidence_interval(n_obs, alpha)
    n = int(np.sum(n_obs))
    assert isinstance(tau_set, list)
    assert all(abs(round(t * n) - t * n) < 1e-9 for t in tau_set)


def test_rigdon_hudgens_nonempty_for_basic_case():
    n_obs = np.array([2, 6, 8, 0])
    alpha = 0.05
    tau_set = Rigdon_Hudgens(n_obs, alpha)
    assert isinstance(tau_set, list)
    assert len(tau_set) > 0

