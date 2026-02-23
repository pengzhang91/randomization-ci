import numpy as np

def next_power_of_two_at_least(x: int) -> int:
    """Return smallest power of two >= x (x>0)."""
    n = 1
    while n < x:
        n <<= 1
    return n

def compute_p_v_fft(
    m: int,
    l: int,
    Delta: float,
    eps_round: float = 1e-12,
    verify_dft: bool = False,
):
    """
    Compute p(v) = Pr(|S| >= |Delta|) where S is described in Section 4.

    Args:
      m: number of ±1 Rademacher terms (v11)
      l: number of ±1/2 half-Rademacher terms (v10 + v01)
      Delta: threshold > 0
      eps_round: numeric tolerance for clipping small negative numerical errors.
      verify_dft: if True, run an explicit inverse-DFT consistency check
        (O(N^2), intended for debugging only).

    Returns:
      p: the tail probability p(v)
    """
    if not isinstance(m, (int, np.integer)) or int(m) < 0:
        raise ValueError(f"m must be a nonnegative integer, got {m}.")
    if not isinstance(l, (int, np.integer)) or int(l) < 0:
        raise ValueError(f"l must be a nonnegative integer, got {l}.")
    if eps_round <= 0:
        raise ValueError(f"eps_round must be positive, got {eps_round}.")
    m = int(m)
    l = int(l)

    M = 2*m + l
    if M == 0:
        return 1.0 if Delta == 0 else 0.0

    N = next_power_of_two_at_least(2*M + 1)
    j = np.arange(N)
    thetas = 4.0 * np.pi * j / N
    F = (np.cos(thetas) ** m) * (np.cos(thetas / 2.0) ** l)

    phase = np.exp(1j * 2.0 * np.pi * j * M / N)
    A = F * phase

    # inverse DFT to recover a_k (k = 0..N-1)
    a = np.fft.fft(A) / N
    if verify_dft:
        # Expensive sanity check: explicit inverse DFT definition.
        k = np.arange(N)
        expo = np.exp(-2j * np.pi * np.outer(j, k) / N)
        a_check = (A[:, None] * expo).sum(axis=0) / N
        if not np.allclose(a, a_check, rtol=1e-10, atol=1e-12):
            raise ValueError("Inverse DFT check failed: a does not match explicit formula")
    
    q_raw = np.real(a[:2*M + 1])

    # Clip small negative values caused by numerical error
    q_raw = np.where(q_raw < 0, np.where(q_raw > -eps_round, 0.0, q_raw), q_raw)
    if np.any(q_raw < 0):
        raise ValueError("q_raw contains negative entries after clipping")

    # Normalize (safe-guard)
    total = q_raw.sum()
    if not np.isclose(total, 1.0, rtol=0.0, atol=eps_round):
        raise ValueError(f"q_raw total probability not 1: total={total}")
    q = q_raw / total

    r_values = np.arange(-M, M + 1, dtype=int)
    support_points = np.abs(r_values / 2.0)
    mask = support_points >= abs(Delta)

    p = float(q[mask].sum())

    return p
