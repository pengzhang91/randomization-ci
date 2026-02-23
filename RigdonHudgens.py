import numpy as np
from fft import compute_p_v_fft

"""
A variant of Rigdon and Hudgens' method for complete randomization.
Specifically, we enumerate all compatible outcome counts.
"""
    
def Rigdon_Hudgens(
    n_obs,
    alpha: float,
    verbose: bool = False,
):
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    n_obs = np.asarray(n_obs)
    if n_obs.shape != (4,):
        raise ValueError(f"n_obs must have shape (4,), got {n_obs.shape}.")
    if not np.issubdtype(n_obs.dtype, np.integer):
        if not np.all(np.equal(n_obs, np.rint(n_obs))):
            raise ValueError("n_obs entries must be integers.")
        n_obs = np.rint(n_obs).astype(int)
    else:
        n_obs = n_obs.astype(int, copy=False)
    if np.any(n_obs < 0):
        raise ValueError("n_obs entries must be nonnegative.")
    n = int(np.sum(n_obs))
    if n <= 0:
        raise ValueError("Total sample size must be positive.")
    tau_set = []

    p_v_calls = 0
    # enumerate all compatible potential outcome count vectors
    for a1 in range(0, n_obs[0]+1):
        a2 = n_obs[0] - a1
        for a3 in range(0, n_obs[1]+1):
            for a5 in range(0, n_obs[2]+1):
                a6 = n_obs[2] - a5
                for a7 in range(0, n_obs[3]+1):
                    v11 = a1 + a5
                    v10 = a2 + a7
                    v01 = a3 + a6
                    tau_0 = (v10 - v01) / n
                    
                    Delta = n_obs[0] - n_obs[2] - tau_0 * (n/2)
                    p = compute_p_v_fft(v11, v10+v01, Delta)
                    p_v_calls += 1
                    if p >= alpha:                     
                        tau_set.append(tau_0)

    if verbose:
        print("RH p_v_calls: ", p_v_calls)
    
    return sorted(set(tau_set))
