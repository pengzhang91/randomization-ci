import numpy as np
from fft import compute_p_v_fft

def _compute_p_v_fft_int(a, b, Delta):
    return compute_p_v_fft(int(a), int(b), Delta)

def compute_delta(n_obs, tau_0: float) -> float:
    n_obs = np.asarray(n_obs)
    n = int(np.sum(n_obs))
    Delta = n_obs[0] - n_obs[2] - tau_0 * (float(n) / 2.0)
    return Delta

def compute_ab(n_obs_02, U, n, Ub, Lb, n_tau_int):
    a = min(n_obs_02, U - n_obs_02, n - Lb, (U - Lb) // 2)
    b = min(U - 2*a, n - a, Ub)
    if (b - n_tau_int) % 2 != 0:
        b = b - 1
    if a + b < n_obs_02:
        a = a - 1
        b = b + 2
    return a, b

def compute_p_max_tau_v11(
    n: int,
    n_obs,
    tau_0: float,
) -> float:
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}.")
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
    if int(np.sum(n_obs)) != int(n):
        raise ValueError(f"sum(n_obs) must equal n, got sum(n_obs)={int(np.sum(n_obs))}, n={n}.")

    Delta = compute_delta(n_obs, tau_0)
    
    n_tau = n * tau_0
    n_tau_int = int(round(n_tau))
    n_obs_02 = n_obs[0] + n_obs[2]

    u1 = 2*n - n_tau - 2*n_obs[1]
    u2 = 2*n + n_tau - 2*n_obs[3]
    U = int(np.floor(min(u1, u2)))
    l1 = 2*n_obs[0] - n_tau
    l2 = 2*n_obs[2] + n_tau
    L = int(np.ceil(max(l1, l2)))

    Ub = min(2*(n_obs[0] + n_obs[3]) - n_tau, 2*(n_obs[1] + n_obs[2]) + n_tau)
    
    if abs(n_tau_int) > 0:
        a, b = compute_ab(n_obs_02, U, n, Ub, abs(n_tau_int), n_tau_int)
        p = _compute_p_v_fft_int(a, b, Delta)
        return p

    ab_set = []
    # case: b = 0
    a = min(n_obs_02, U - n_obs_02, n, U // 2)
    if a >= n_obs_02 and 2*a >= L:
        ab_set.append((a, 0))
    # case: b not 0
    a, b = compute_ab(n_obs_02,  U, n, Ub, 2, n_tau_int)
    if a >= 0 and a+b >= n_obs_02 and 2*a+b >= L:
        ab_set.append((a, b))

    p_set = [_compute_p_v_fft_int(a_i, b_i, Delta) for a_i, b_i in ab_set]
    return max(p_set) if p_set else 0.0


if __name__ == "__main__":
    n_obs = np.array([8, 2, 2, 8])
    n = int(np.sum(n_obs))
    tau_0 = 0.0

    compute_p_max_tau_v11(n, n_obs, tau_0)
