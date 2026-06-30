from fractions import Fraction

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

def _validate_count(x, name: str) -> int:
    if not isinstance(x, (int, np.integer)) or int(x) < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {x}.")
    return int(x)

def _probability_fraction(
    p,
    max_denominator: int,
    p_tol: float,
) -> Fraction:
    if not isinstance(max_denominator, (int, np.integer)) or int(max_denominator) <= 0:
        raise ValueError(
            f"max_denominator must be a positive integer, got {max_denominator}."
        )
    if p_tol < 0:
        raise ValueError(f"p_tol must be nonnegative, got {p_tol}.")

    if isinstance(p, Fraction):
        p_frac = p
    elif isinstance(p, str):
        p_frac = Fraction(p)
    else:
        p_float = float(p)
        p_frac = Fraction(p_float).limit_denominator(int(max_denominator))
        if abs(float(p_frac) - p_float) > p_tol:
            raise ValueError(
                "p is not close to a rational with denominator at most "
                f"{max_denominator}: got {p}."
            )

    if not (0 < p_frac < 1):
        raise ValueError(f"p must be in (0, 1), got {p}.")
    return p_frac

def compute_s_v_p_distribution_fft(
    v11: int,
    v10: int,
    v01: int,
    p,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
    verify_dft: bool = False,
):
    """
    Compute the distribution of S_{v,p} from page 19 for rational p.

    Page 19 writes

      S_{v,p} = sum_{r=1}^{v11} xi_r
              + sum_{s=1}^{v10} zeta_s
              + sum_{t=1}^{v01} eta_t,

    where

      xi   is  1/p with probability p and -1/(1-p) with probability 1-p,
      zeta is (1-p)/p with probability p and -1 with probability 1-p,
      eta  is  1 with probability p and -p/(1-p) with probability 1-p.

    v00 does not affect S_{v,p}, so it is not an argument.

    Args:
      v11, v10, v01: nonnegative integer potential-outcome counts.
      p: treatment probability. Use Fraction or a decimal/rational value with
        denominator <= max_denominator.
      max_denominator: denominator bound used when converting float p.
      p_tol: tolerance for accepting the rational approximation to float p.
      eps_round: numeric tolerance for clipping small negative FFT errors.
      verify_dft: if True, run an explicit inverse-DFT consistency check
        (O(N^2), intended for debugging only).

    Returns:
      support: float array of S_{v,p} support points.
      q: probability mass at the corresponding support points.
    """
    v11 = _validate_count(v11, "v11")
    v10 = _validate_count(v10, "v10")
    v01 = _validate_count(v01, "v01")
    if eps_round <= 0:
        raise ValueError(f"eps_round must be positive, got {eps_round}.")

    p_frac = _probability_fraction(p, max_denominator, p_tol)
    a = p_frac.numerator
    d = p_frac.denominator
    b = d - a

    if v11 + v10 + v01 == 0:
        return np.array([0.0]), np.array([1.0])

    scale = a * b

    low = -(v11 * d * a + v10 * a * b + v01 * a * a)
    high = v11 * d * b + v10 * b * b + v01 * a * b
    degree = high - low

    step11 = d * d
    step10 = b * d
    step01 = a * d

    N = next_power_of_two_at_least(degree + 1)
    j = np.arange(N)
    theta = 2.0 * np.pi * j / N
    p_float = float(p_frac)
    q_float = 1.0 - p_float

    F = (
        (q_float + p_float * np.exp(1j * theta * step11)) ** v11
        * (q_float + p_float * np.exp(1j * theta * step10)) ** v10
        * (q_float + p_float * np.exp(1j * theta * step01)) ** v01
    )

    coeff = np.fft.fft(F) / N
    if verify_dft:
        k = np.arange(N)
        expo = np.exp(-2j * np.pi * np.outer(j, k) / N)
        coeff_check = (F[:, None] * expo).sum(axis=0) / N
        if not np.allclose(coeff, coeff_check, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Inverse DFT check failed: coefficients do not match explicit formula"
            )

    q_raw = np.real(coeff[:degree + 1])
    q_raw = np.where(q_raw < 0, np.where(q_raw > -eps_round, 0.0, q_raw), q_raw)
    if np.any(q_raw < 0):
        raise ValueError("q_raw contains negative entries after clipping")

    total = q_raw.sum()
    if not np.isclose(total, 1.0, rtol=0.0, atol=eps_round):
        raise ValueError(f"q_raw total probability not 1: total={total}")
    q = q_raw / total

    scaled_support = low + np.arange(degree + 1, dtype=int)
    support = scaled_support / float(scale)

    return support, q

def compute_p_v_p_fft(
    v11: int,
    v10: int,
    v01: int,
    p,
    Delta: float,
    max_denominator: int = 10_000,
    p_tol: float = 1e-12,
    eps_round: float = 1e-10,
    verify_dft: bool = False,
):
    """
    Compute p^{Ber(p)}(v, Delta) = Pr(|S_{v,p}| >= |Delta|).

    This is the page-19 general Bernoulli analogue of compute_p_v_fft.
    The v00 count is omitted because it contributes only zeros to S_{v,p}.
    """
    support, q = compute_s_v_p_distribution_fft(
        v11,
        v10,
        v01,
        p,
        max_denominator=max_denominator,
        p_tol=p_tol,
        eps_round=eps_round,
        verify_dft=verify_dft,
    )
    threshold = abs(float(Delta))
    tol = eps_round * max(1.0, threshold)
    return float(q[np.abs(support) >= threshold - tol].sum())
