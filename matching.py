from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import time
from scipy.stats import norm
from fft import compute_p_v_fft
from search_utils import confidence_interval_from_candidates

try:
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, ProblemStatus
    _MOSEK_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    Model = Domain = Expr = ObjectiveSense = ProblemStatus = None  # type: ignore[assignment]
    _MOSEK_IMPORT_ERROR = exc


W_VALS = [-1, 0, 1]
U_VALS = [-1, 0, 1]
WZ_KEYS = [(w, z) for w in W_VALS for z in (0, 1)]

# Flatten (w,u) -> index in a 9-vector, row-major by w then u.
IDX = {(w, u): 3 * i + j for i, w in enumerate(W_VALS) for j, u in enumerate(U_VALS)}


@dataclass(frozen=True)
class LexiResult:
    m2: int
    m1: int
    v: Dict[Tuple[int, int], int]  # v[(w,u)]


class InfeasibleOptimizationError(RuntimeError):
    """Raised when a Mosek model is infeasible."""


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def _validate_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer or None, got {type(seed)}.")


def _validate_n_wz_dict(N_wz) -> Dict[Tuple[int, int], int]:
    if not isinstance(N_wz, dict):
        raise ValueError("N_wz must be a dict keyed by (w, z).")
    out = {}
    for key in WZ_KEYS:
        val = N_wz.get(key, 0)
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"N_wz[{key}] must be an integer, got {type(val)}.")
        if int(val) < 0:
            raise ValueError(f"N_wz[{key}] must be nonnegative, got {val}.")
        out[key] = int(val)
    if sum(out.values()) <= 0:
        raise ValueError("Total matched-pair count must be positive.")
    return out


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


def _get_problem_status_name(M: Model) -> str:
    try:
        status = M.getProblemStatus()
    except Exception:
        return "unknown"
    return str(status)


def _is_infeasible_status(M: Model) -> bool:
    if ProblemStatus is None:
        return False
    try:
        status = M.getProblemStatus()
    except Exception:
        return False
    return status in {
        ProblemStatus.PrimalInfeasible,
        ProblemStatus.DualInfeasible,
        ProblemStatus.PrimalAndDualInfeasible,
    }


def _solve_model_and_get_v(M: Model, vvar) -> Dict[Tuple[int, int], int]:
    if _MOSEK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mosek is required for optimization. Install Mosek's Python package and "
            "ensure a valid license is available."
        ) from _MOSEK_IMPORT_ERROR
    try:
        M.solve()
    except Exception as exc:
        msg = str(exc).lower()
        if "infeasible" in msg:
            raise InfeasibleOptimizationError(
                f"Model infeasible during solve ({_get_problem_status_name(M)})."
            ) from exc
        raise RuntimeError(
            f"Mosek solve failed ({_get_problem_status_name(M)}): {exc}"
        ) from exc

    if _is_infeasible_status(M):
        raise InfeasibleOptimizationError(
            f"Model infeasible after solve ({_get_problem_status_name(M)})."
        )

    sol = vvar.level()  # length 9
    # Round defensively (Fusion returns floats); they should be integral at optimality.
    sol_int = np.rint(sol).astype(int)
    return {(w, u): int(sol_int[IDX[(w, u)]]) for w in W_VALS for u in U_VALS}


def _build_base_model(N_w: Dict[int, int], rhs_ate: int) -> Tuple[Model, any]:
    """
    Builds the feasible set V in Sec 4.2.3:
      (10) sum_u v_{w,u} = N_w for w in {-1,0,1}
      (11) sum_{w,u} u * v_{w,u} = rhs_ate  (rhs_ate = n*tau0 - S_obs)
    with v_{w,u} >= 0 and integer.
    """
    if _MOSEK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mosek is required for optimization. Install Mosek's Python package and "
            "ensure a valid license is available."
        ) from _MOSEK_IMPORT_ERROR
    M = Model("sec_4_2_3_base")
    v = M.variable("v", 9, Domain.integral(Domain.greaterThan(0.0)))

    # (10) Row sums: sum_u v_{w,u} = N_w
    for w in W_VALS:
        inds = [IDX[(w, u)] for u in U_VALS]
        M.constraint(f"row_sum_w={w}", Expr.sum(v.pick(inds)), Domain.equalsTo(float(N_w[w])))

    # (11) ATE compatibility: sum_{w,u} u * v_{w,u} = rhs_ate
    coeff_u = np.array([u for w in W_VALS for u in U_VALS], dtype=float)
    M.constraint("ate_constraint", Expr.dot(coeff_u, v), Domain.equalsTo(float(rhs_ate)))

    return M, v


def find_lexicographically_largest_m2_m1(
    N_wz: Dict[Tuple[int, int], int],
    tau0: float,
    S_obs: int,
    n: int,
) -> Tuple[Optional[LexiResult], Optional[LexiResult]]:
    """
    Implements Sec 4.2.3:

    - Build V via (10)-(11)
    - Solve ILP for M0 (m1 = 0): maximize m2
    - Solve ILP for M1 (m1 >= 1): maximize m2, then (fixing optimal m2 components) maximize m1

    Returns:
      (best_in_M0, best_in_M1) where each is a LexiResult or None if infeasible/empty.
    """

    # N_w := N_{w,0} + N_{w,1}
    N_w = {w: int(N_wz.get((w, 0), 0) + N_wz.get((w, 1), 0)) for w in W_VALS}

    # rhs in (11): n*tau0 - S_obs must be integer in the paper's grid; round defensively.
    rhs_ate = int(round(n * tau0 - S_obs))

    # m2 = v_{1,-1} + v_{-1,1}
    m2_inds = [IDX[(1, -1)], IDX[(-1, 1)]]
    # m1 = v_{1,0} + v_{0,1} + v_{0,-1} + v_{-1,0}
    m1_inds = [IDX[(1, 0)], IDX[(0, 1)], IDX[(0, -1)], IDX[(-1, 0)]]

    # -------------------------
    # ILP for M0: m1 == 0
    # max m2 s.t. v in V and m1 == 0
    # -------------------------
    best_M0: Optional[LexiResult] = None
    try:
        with _build_base_model(N_w, rhs_ate)[0] as M:
            v = M.getVariable("v")  # retrieve by name from the model context

            M.constraint("m1_eq_0", Expr.sum(v.pick(m1_inds)), Domain.equalsTo(0.0))
            M.objective("max_m2", ObjectiveSense.Maximize, Expr.sum(v.pick(m2_inds)))

            vsol = _solve_model_and_get_v(M, v)
            m2 = vsol[(1, -1)] + vsol[(-1, 1)]
            m1 = vsol[(1, 0)] + vsol[(0, 1)] + vsol[(0, -1)] + vsol[(-1, 0)]
            best_M0 = LexiResult(m2=m2, m1=m1, v=vsol)
    except InfeasibleOptimizationError:
        # Infeasible M0. Treat as empty.
        best_M0 = None

    # -------------------------
    # ILP(s) for M1: m1 >= 1
    # Step 1: max m2 s.t. v in V and m1 >= 1
    # Step 2: fix v_{1,-1} + v_{-1,1} at the step-1 optimum, then max m1
    # -------------------------
    best_M1: Optional[LexiResult] = None
    try:
        # Step 1
        with _build_base_model(N_w, rhs_ate)[0] as M1:
            v1 = M1.getVariable("v")

            M1.constraint("m1_ge_1", Expr.sum(v1.pick(m1_inds)), Domain.greaterThan(1.0))
            M1.objective("max_m2", ObjectiveSense.Maximize, Expr.sum(v1.pick(m2_inds)))

            vsol1 = _solve_model_and_get_v(M1, v1)
            m2_star = vsol1[(1, -1)] + vsol1[(-1, 1)]

        # Step 2 (new model): fix optimal m2, maximize m1
        with _build_base_model(N_w, rhs_ate)[0] as M2:
            v2 = M2.getVariable("v")

            M2.constraint("fix_m2", Expr.sum(v2.pick(m2_inds)), Domain.equalsTo(float(m2_star)))
            # Optional: keep m1 >= 1 to stay in M1
            M2.constraint("m1_ge_1", Expr.sum(v2.pick(m1_inds)), Domain.greaterThan(1.0))

            M2.objective("max_m1", ObjectiveSense.Maximize, Expr.sum(v2.pick(m1_inds)))

            vsol2 = _solve_model_and_get_v(M2, v2)
            m2 = vsol2[(1, -1)] + vsol2[(-1, 1)]
            m1 = vsol2[(1, 0)] + vsol2[(0, 1)] + vsol2[(0, -1)] + vsol2[(-1, 0)]
            best_M1 = LexiResult(m2=m2, m1=m1, v=vsol2)

    except InfeasibleOptimizationError:
        best_M1 = None

    return best_M0, best_M1

def compute_p_max(
    n: int,
    N_wz,
    S_obs,
    tau0: float,
) -> float:
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    res_M0, res_M1 = find_lexicographically_largest_m2_m1(N_wz, tau0, S_obs, n)
    Delta = abs(2*S_obs - tau0*n)
    ps = []
    if res_M0 is not None:
        ps.append(compute_p_v_fft(res_M0.m2, res_M0.m1, Delta / 2))
    if res_M1 is not None:
        ps.append(compute_p_v_fft(res_M1.m2, res_M1.m1, Delta / 2))
    if not ps:
        return float("nan")
    return max(ps)


def permutation_based_confidence_interval_match(
    N_wz,
    alpha: float,
    verbose: bool = False,
):
    _ = verbose
    _validate_alpha(alpha)
    N_wz = _validate_n_wz_dict(N_wz)
    m = int(np.sum(list(N_wz.values())))
    n = m * 2
    
    S_obs = -(
        N_wz.get((-1, 0), 0) + N_wz.get((-1, 1), 0)
    ) + (
        N_wz.get((1, 0), 0) + N_wz.get((1, 1), 0)
    )

    taus = np.arange(S_obs - m, S_obs + m + 1) / n   # set of ATE candidates
    K = len(taus)
    if K == 0:
        return []
    
    # observed test statistic T_obs = the HT estimator
    T_obs = S_obs * (2.0 / n)
    
    def p_max_func(tau):
        return compute_p_max(n, N_wz, S_obs, tau)
    return confidence_interval_from_candidates(taus, T_obs, alpha, p_max_func)
    

def wald_ci_match(
    N_wz,
    alpha: float = 0.05,
    zcrit: float = 1.96,
):
    _validate_alpha(alpha)
    N_wz = _validate_n_wz_dict(N_wz)
    m = int(np.sum(list(N_wz.values())))
    if m < 2:
        raise ValueError("Need at least 2 matched pairs for Wald interval.")
    n = m * 2

    S_obs = -(
        N_wz.get((-1, 0), 0) + N_wz.get((-1, 1), 0)
    ) + (
        N_wz.get((1, 0), 0) + N_wz.get((1, 1), 0)
    )

    # the HT estimator
    tau_hat = 2.0 * S_obs / n

    ss = (
        (-1.0 - tau_hat) ** 2 * (N_wz.get((-1,0),0) + N_wz.get((-1,1),0)) + 
        (1.0 - tau_hat) ** 2 * (N_wz.get((1,0),0) + N_wz.get((1,1),0)) + 
        (0.0 - tau_hat) ** 2 * (N_wz.get((0,0),0) + N_wz.get((0,1),0))
    )
    se = np.sqrt(ss / (m * (m-1)))

    return (tau_hat - zcrit * se, tau_hat + zcrit * se)

def ci_test_match(
    v, # 6 entries
    alpha: float = 0.05,
    its: int = 1000,
    seed: Optional[int] = 123,
):
    """
    Monte Carlo comparison of Wald and exact matched-pair CIs.

    Args:
      v = [v_{1,1}, v_{1,0}, v_{1,-1}, v_{0,0}, v_{0,-1}, v_{-1,-1}]: 
        length-6 nonnegative integer pairlevel potential-outcome counts. v_{y,y'} and v_{y',y} is symmetric and thus we omit one.
      alpha: nominal type-I error level.
      its: number of Monte Carlo draws.
      seed: RNG seed (None for nondeterministic entropy).
    """
    _validate_alpha(alpha)
    if its <= 0:
        raise ValueError(f"its must be positive, got {its}.")
    _validate_seed(seed)
    v = _validate_v6(v)
    n = int(np.sum(v) * 2)

    tau = (2.0*v[0] + v[1] - v[4] -2.0*v[5]) / n

    print("v: ", v, "tau: ", tau, "seed: ", seed)
    
    lo_wald = np.empty(its, dtype=float)
    hi_wald = np.empty(its, dtype=float)
    lo_exact = np.full(its, np.nan, dtype=float)
    hi_exact = np.full(its, np.nan, dtype=float)
    
    rng = np.random.default_rng(seed)
    zcrit = norm.ppf(1 - alpha/2)
    x_all = rng.binomial(v, 0.5, size=(its, 6))
    start = time.time()
    for i in range(its):
        if i % 1000 == 0:
            elapsed = time.time() - start
            print(f"trial {i}, elapsed {elapsed:.2f}s")

        x = x_all[i]
        N_wz = {
            (-1, 0): v[2]-x[2] + v[4]-x[4] + v[5]-x[5], 
            (-1, 1): x[5],
            (0, 0): v[1]-x[1] + v[3]-x[3],  
            (0, 1): x[3] + x[4],
            (1, 0): v[0]-x[0],
            (1, 1): x[0] + x[1] + x[2],
        }
        
        lo, hi = wald_ci_match(N_wz, alpha, zcrit=zcrit)
        lo_wald[i] = lo
        hi_wald[i] = hi
        
        tau_set = permutation_based_confidence_interval_match(N_wz, alpha)
        if not tau_set:
            # print(f"empty tau_set for n_obs={n_obs}")
            continue
        lo = min(tau_set)
        hi = max(tau_set)
        lo_exact[i] = lo
        hi_exact[i] = hi

    widths_wald = hi_wald - lo_wald
    widths_exact = hi_exact - lo_exact
    cover_wald = np.count_nonzero((lo_wald <= tau) & (tau <= hi_wald))
    cover_exact = np.count_nonzero((lo_exact <= tau) & (tau <= hi_exact))

    result = (
        np.median(widths_wald),
        np.nanmedian(widths_exact),
        100.0 * cover_wald / its,
        100.0 * cover_exact / its,
    )
    print(
        "Median width (Wald):", result[0],
        "Median width (Exact):", result[1],
        "Coverage % (Wald):", result[2],
        "Coverage % (Exact):", result[3],
    )
    
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v", nargs=6, type=int, required=True, help="6 counts")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--its", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    result = ci_test_match(args.v, alpha=args.alpha, its=args.its, seed=args.seed)
    # print(result)
