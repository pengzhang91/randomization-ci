import numpy as np
from compute_pmax import compute_p_max_tau_v11
from RigdonHudgens import Rigdon_Hudgens
from search_utils import confidence_interval_from_candidates


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

def randomization_based_confidence_interval(
    n_obs,
    alpha: float,
    verbose: bool = False,
):
    _ = verbose
    _validate_alpha(alpha)
    n_obs = _validate_n_obs(n_obs)
    
    n1 = n_obs[0] + n_obs[1]
    S = n_obs[0] - n_obs[2]
    n = int(np.sum(n_obs))
    taus = np.arange(S - n1, S - n1 + n + 1) / n   # set of ATE candidates
    K = len(taus)
    if K == 0:
        return []
    
    # observed test statistic T_obs = the HT estimator
    T_obs = S * (2 / n)
    
    def p_max_func(tau):
        return compute_p_max_tau_v11(n, n_obs, tau)
    return confidence_interval_from_candidates(taus, T_obs, alpha, p_max_func)
    

# Example usage
if __name__ == "__main__":
    alpha = 0.05

    # test examples
    # observation table (n11, n10, n01, n00), n_{zy}
    n_obs = np.array([2, 6, 8, 0])
    # n_obs = np.array([6, 4, 4, 6])
    # n_obs = np.array([8, 4, 5, 7])
    # n_obs = np.array([10, 13, 15, 12])
    # n_obs = np.array([0, 6, 1, 5])
    # n_obs = np.array([6, 0, 6, 0])
    # n_obs = np.array([1, 5, 5, 1])
    # n_obs = np.array([5, 1, 0, 6])
    # n_obs = np.array([10, 8, 0, 2])
    
    n = np.sum(n_obs)

    print("observed data: ", n_obs)

    tau_set = randomization_based_confidence_interval(n_obs, alpha, verbose=True)
    tau_set_rh = Rigdon_Hudgens(n_obs, alpha, verbose=True)

    tau_set = [int(t * n) for t in tau_set]
    tau_set_rh = [int(t * n) for t in tau_set_rh]
    print("my confidence set (multiplied by n): ", tau_set)
    print("RH confidence set (multiplied by n): ", tau_set_rh)

    diff = len(set(tau_set).symmetric_difference(tau_set_rh))
    print("difference between my set and RH set: ", diff)
