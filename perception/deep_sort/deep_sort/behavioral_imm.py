"""
behavioral_imm.py  –  Interacting Multiple Model (IMM) filter for pedestrian tracking.

Drop-in replacement for BehavioralEKFFilter (behavioral_ekf_2.py).

Blends three kinematic sub-filters:
  Model 0 (CV) : Constant Velocity  – steady walking, no turn
  Model 1 (CT) : Coordinated Turn   – turning / swerving pedestrian
  Model 2 (CA) : Constant Accel/Stop – yielding or stopping

Internal EKF state per model (5-D):  [px, py, v, phi, omega]
External DeepSORT state       (8-D):  [x, y, a, h, vx, vy, va, vh]

Packed state layout (26-D mean, 26×26 covariance):
  mean[ 0: 8]  – fused DeepSORT state
  mean[ 8:13]  – CV model internal state
  mean[13:18]  – CT model internal state
  mean[18:23]  – CA model internal state
  mean[23:26]  – mode probabilities  [μ_cv, μ_ct, μ_ca]

  cov[ 0:8,  0:8]  – fused 8×8 covariance
  cov[ 8:13, 8:13] – CV  5×5 covariance
  cov[13:18,13:18] – CT  5×5 covariance
  cov[18:23,18:23] – CA  5×5 covariance

NOTE: track.py inflates covariance by 1.5^N during occlusions.  When using
IMM this aggressive scalar should be reduced (e.g. 1.05^N) or removed, as
the process noise Q in each sub-filter already propagates uncertainty forward
in a principled way.  The inflation still works mechanically (it scales all
26×26 blocks) but may over-widen gating after long occlusions.
"""

import warnings
import numpy as np
import scipy.linalg

from . import kalman_filter as _kf

# Suppress scipy's LinAlgWarning for ill-conditioned matrices — we handle these
# explicitly with trace-relative regularisation throughout this module.
warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)

chi2inv95 = _kf.chi2inv95  # re-export for tracker.py gating

# ── Model index aliases ────────────────────────────────────────────────────────
CV_IDX, CT_IDX, CA_IDX = 0, 1, 2
_N = 3  # number of models

# ── State packing dimensions ───────────────────────────────────────────────────
_DIM_FUSED = 8
_DIM_INNER = 5
_DIM_PROB  = _N
_TOTAL     = _DIM_FUSED + _N * _DIM_INNER + _DIM_PROB  # 26

_SL_FUSED = slice(0, 8)
_SL_M     = [slice(8 + i * 5, 8 + (i + 1) * 5) for i in range(_N)]  # [8:13, 13:18, 18:23]
_SL_PROB  = slice(23, 26)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _vphi_from_xy(vx: float, vy: float):
    v = np.hypot(vx, vy)
    phi = float(np.arctan2(vy, vx)) if v > 1e-8 else 0.0
    return v, phi


def _xy_from_vphi(v: float, phi: float):
    return v * np.cos(phi), v * np.sin(phi)


def _nearest_psd(M: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite matrix
    via eigenvalue clamping (Higham 1988).

    After reconstruction via V @ diag(λ) @ Vᵀ, floating-point rounding can
    leave eigenvalues at ~-1e-16.  Adding min_eig·I guarantees strict PD so
    that np.linalg.cholesky always succeeds.
    """
    B = (M + M.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, min_eig)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Explicit diagonal bump to survive float reconstruction rounding
    result += np.eye(len(result)) * min_eig
    return result


def _log_gaussian_likelihood(innov: np.ndarray, S: np.ndarray) -> float:
    """Log-likelihood of innovation ∈ ℝⁿ under N(0, S).

    Uses Cholesky to avoid explicit matrix inversion.
    Returns -inf on numerical failure (degenerate S).
    """
    n = innov.shape[0]
    try:
        chol = np.linalg.cholesky(S)
        z = scipy.linalg.solve_triangular(chol, innov, lower=True, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (n * np.log(2.0 * np.pi) + log_det + float(z @ z))
    except np.linalg.LinAlgError:
        return -1e10


# ══════════════════════════════════════════════════════════════════════════════
#  Per-model state transition functions  (operate on 5-D internal state)
# ══════════════════════════════════════════════════════════════════════════════

def _cv_predict(x5: np.ndarray, P5: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray):
    """Constant Velocity prediction step.

    State transition (ω forced to 0, v and φ held constant):
        px_new = px + v·cos(φ)·dt
        py_new = py + v·sin(φ)·dt

    Social force is injected as a kinematic control input:
        px_new += ½·f_x·dt²,  py_new += ½·f_y·dt²
    """
    px, py, v, phi, _ = x5

    x_pred = np.array([
        px + v * np.cos(phi) * dt + 0.5 * sf[0] * dt**2,
        py + v * np.sin(phi) * dt + 0.5 * sf[1] * dt**2,
        v,
        phi,
        0.0,   # CV clamps turn rate to zero
    ])

    # Jacobian ∂f/∂x  (5×5)
    F = np.eye(5)
    F[0, 2] =  np.cos(phi) * dt
    F[0, 3] = -v * np.sin(phi) * dt
    F[1, 2] =  np.sin(phi) * dt
    F[1, 3] =  v * np.cos(phi) * dt
    # ∂/∂ω = 0 for CV (row 0,1 col 4 stay zero)

    P_pred = F @ P5 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


def _ct_predict(x5: np.ndarray, P5: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray):
    """Coordinated Turn prediction step.

    Ports the exact Jacobian from behavioral_ekf_2.py, including the
    ∂/∂ω column that was missing in the original behavioral_ekf.py.

    State transition (arc or straight-line when |ω| < 1e-4):
        For |ω| ≥ 1e-4:
            px_new = px + (v/ω)·(sin(φ + ω·dt) − sin(φ))
            py_new = py + (v/ω)·(cos(φ)        − cos(φ + ω·dt))
        φ_new = φ + ω·dt
    """
    px, py, v, phi, omega = x5

    x_pred = x5.copy()
    if abs(omega) < 1e-4:
        x_pred[0] = px + v * np.cos(phi) * dt
        x_pred[1] = py + v * np.sin(phi) * dt
    else:
        x_pred[0] = px + (v / omega) * (np.sin(phi + omega * dt) - np.sin(phi))
        x_pred[1] = py + (v / omega) * (np.cos(phi) - np.cos(phi + omega * dt))
    x_pred[3] = phi + omega * dt
    # apply social force
    x_pred[0] += 0.5 * sf[0] * dt**2
    x_pred[1] += 0.5 * sf[1] * dt**2

    # Full Jacobian (including ∂/∂ω terms)
    F = np.eye(5)
    if abs(omega) < 1e-4:
        F[0, 2] =  np.cos(phi) * dt
        F[0, 3] = -v * np.sin(phi) * dt
        F[0, 4] = -0.5 * v * np.sin(phi) * dt**2
        F[1, 2] =  np.sin(phi) * dt
        F[1, 3] =  v * np.cos(phi) * dt
        F[1, 4] =  0.5 * v * np.cos(phi) * dt**2
    else:
        s_phi     = np.sin(phi)
        c_phi     = np.cos(phi)
        s_phi_odt = np.sin(phi + omega * dt)
        c_phi_odt = np.cos(phi + omega * dt)

        F[0, 2] = (s_phi_odt - s_phi) / omega
        F[0, 3] = (v / omega) * (c_phi_odt - c_phi)
        F[1, 2] = (c_phi - c_phi_odt) / omega
        F[1, 3] = (v / omega) * (s_phi_odt - s_phi)
        F[0, 4] = v * (omega * dt * c_phi_odt - s_phi_odt + s_phi) / omega**2
        F[1, 4] = v * (omega * dt * s_phi_odt + c_phi_odt - c_phi) / omega**2
    F[3, 4] = dt  # ∂φ/∂ω

    P_pred = F @ P5 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


def _ca_predict(x5: np.ndarray, P5: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray, gamma: float = 0.85):
    """Constant Deceleration / Stop prediction step.

    Models a pedestrian yielding or slowing to a halt.  Uses CV geometry but
    applies a velocity damping factor γ ∈ (0, 1):
        px_new = px + v·cos(φ)·dt   (displacement uses current v)
        v_new  = γ · v              (speed decays each timestep)

    The Jacobian captures this damping via F[2, 2] = γ.
    A large process noise Q[2, 2] lets the filter recover if the pedestrian
    re-accelerates.
    """
    px, py, v, phi, _ = x5

    x_pred = np.array([
        px + v * np.cos(phi) * dt + 0.5 * sf[0] * dt**2,
        py + v * np.sin(phi) * dt + 0.5 * sf[1] * dt**2,
        gamma * v,   # damped velocity
        phi,
        0.0,         # no turn in CA
    ])

    F = np.eye(5)
    F[0, 2] =  np.cos(phi) * dt
    F[0, 3] = -v * np.sin(phi) * dt
    F[1, 2] =  np.sin(phi) * dt
    F[1, 3] =  v * np.cos(phi) * dt
    F[2, 2] =  gamma   # velocity damping in Jacobian

    P_pred = F @ P5 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


def _ekf_update_5d(x5: np.ndarray, P5: np.ndarray,
                   z_xy: np.ndarray, R: np.ndarray):
    """EKF measurement update for 5-D state, 2-D position observation [px, py].

    Uses Joseph-form covariance update for numerical stability:
        P_new = (I - KH)·P·(I - KH)ᵀ + K·R·Kᵀ

    Returns (x_new, P_new, innovation, innovation_cov).
    """
    # Observation matrix H: observe only px, py
    H = np.zeros((2, 5))
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    innov = z_xy - H @ x5           # y = z − Hx̂
    S     = (H @ P5 @ H.T + R)      # innovation covariance
    S     = (S + S.T) * 0.5         # symmetrise to kill float drift

    # Trace-relative regularisation on S (same reasoning as gating_distance):
    # when the position sub-block of P5 is large AND highly correlated (common
    # after IMM mixing), det(S) can collapse to ~0 even though S is theoretically
    # PSD.  A fixed epsilon like 1e-8 is negligible at those scales.
    reg_S = max(np.trace(S) * 1e-6, 1e-8)
    S_reg = S + np.eye(2) * reg_S

    # Kalman gain: solve S·Kᵀ = H·P  →  K = P·Hᵀ·S⁻¹
    try:
        K = scipy.linalg.solve(S_reg, H @ P5, assume_a='pos').T
    except np.linalg.LinAlgError:
        # S is still singular (extreme ill-conditioning) — skip update safely.
        return x5.copy(), P5.copy(), innov, S_reg

    x_new = x5 + K @ innov
    I_KH  = np.eye(5) - K @ H
    # Joseph form prevents covariance becoming non-PSD due to float errors
    P_new = I_KH @ P5 @ I_KH.T + K @ R @ K.T

    return x_new, _nearest_psd(P_new), innov, S_reg


# ══════════════════════════════════════════════════════════════════════════════
#  8-D ↔ 5-D bridging helpers  (ported from behavioral_ekf_2.py)
# ══════════════════════════════════════════════════════════════════════════════

def _mean_8_to_5(m8: np.ndarray) -> np.ndarray:
    """Extract [px, py, v, φ, 0] from DeepSORT 8-D state."""
    vx, vy = m8[4], m8[5]
    v, phi = _vphi_from_xy(vx, vy)
    return np.array([m8[0], m8[1], v, phi, 0.0])


def _cov_8_to_5(m8: np.ndarray, c8: np.ndarray) -> np.ndarray:
    """Project 8×8 covariance to 5×5 internal covariance."""
    v, phi = _vphi_from_xy(m8[4], m8[5])
    c5 = np.eye(5) * 1e-4
    c5[0:2, 0:2] = c8[0:2, 0:2]
    # Jacobian: (vx, vy) → (v, φ) via chain rule
    J_inv = np.array([
        [ np.cos(phi),             np.sin(phi)           ],
        [-np.sin(phi) / (v + 1e-8), np.cos(phi) / (v + 1e-8)],
    ])
    c5[2:4, 2:4] = J_inv @ c8[4:6, 4:6] @ J_inv.T
    c5[0:2, 2:4] = c8[0:2, 4:6] @ J_inv.T
    c5[2:4, 0:2] = c5[0:2, 2:4].T
    return _nearest_psd(c5)


def _mean_cov_5_to_8(m5: np.ndarray, c5: np.ndarray,
                     m8_prev: np.ndarray, c8_prev: np.ndarray,
                     dt: float, w_pos: float, w_vel: float):
    """Convert 5-D EKF output back to DeepSORT 8-D state.

    Position and velocity come from the EKF; aspect ratio a and height h are
    propagated with constant-velocity dynamics (a, h sub-filter).
    """
    x, y, v, phi, omega = m5
    vx, vy = _xy_from_vphi(v, phi)
    a  = m8_prev[2] + m8_prev[6] * dt
    h  = m8_prev[3] + m8_prev[7] * dt
    va, vh = m8_prev[6], m8_prev[7]
    m8_new = np.array([x, y, a, h, vx, vy, va, vh])

    # Jacobian for (v, φ) → (vx, vy)
    J = np.eye(8)
    J[4, 2] =  np.cos(phi); J[4, 3] = -v * np.sin(phi)
    J[5, 2] =  np.sin(phi); J[5, 3] =  v * np.cos(phi)

    cov_xy_4 = np.eye(4)
    cov_xy_4[0:2, 0:2] = c5[0:2, 0:2]
    cov_xy_4[2:4, 2:4] = J[4:6, 2:4] @ c5[2:4, 2:4] @ J[4:6, 2:4].T
    cov_xy_4[0:2, 2:4] = c5[0:2, 2:4] @ J[4:6, 2:4].T
    cov_xy_4[2:4, 0:2] = cov_xy_4[0:2, 2:4].T

    # Propagate (a, h, va, vh) sub-filter
    idx_ah   = [2, 3, 6, 7]
    F_ah     = np.eye(4); F_ah[0, 2] = dt; F_ah[1, 3] = dt
    std_ah   = [w_pos * m8_prev[3], w_pos * m8_prev[3], 1e-5, w_vel * m8_prev[3]]
    Q_ah     = np.diag(np.square(std_ah))
    cov_ah_4 = F_ah @ c8_prev[np.ix_(idx_ah, idx_ah)] @ F_ah.T + Q_ah

    c8_new = np.eye(8)
    c8_new[0:4, 0:4] = cov_xy_4
    c8_new[np.ix_(idx_ah, idx_ah)] = cov_ah_4
    return m8_new, _nearest_psd(c8_new)


# ══════════════════════════════════════════════════════════════════════════════
#  Social force
# ══════════════════════════════════════════════════════════════════════════════

def _compute_social_force(x5: np.ndarray, other_xy: list,
                          A: float = 2.0, B: float = 0.5,
                          r: float = 0.3) -> np.ndarray:
    """Exponential repulsion from neighbouring pedestrians.

    f = A · exp((2r − d) / B) · n̂_{ij}   for d < 3 units
    """
    f = np.zeros(2)
    px, py = x5[0], x5[1]
    for nx, ny in other_xy:
        d = np.hypot(nx - px, ny - py)
        if d < 0.01 or d >= 3.0:
            continue
        n_ij = np.array([px - nx, py - ny]) / d
        f += A * np.exp((2.0 * r - d) / B) * n_ij
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  State pack / unpack
# ══════════════════════════════════════════════════════════════════════════════

def _unpack(mean: np.ndarray, cov: np.ndarray):
    """Decompose 26-D mean / 26×26 covariance into IMM components."""
    x_fused  = mean[_SL_FUSED].copy()
    P_fused  = cov[_SL_FUSED, _SL_FUSED].copy()
    x_models = [mean[sl].copy() for sl in _SL_M]
    P_models = [cov[sl, sl].copy()  for sl in _SL_M]
    mu       = mean[_SL_PROB].copy()
    mu       = np.maximum(mu, 1e-8)
    mu      /= mu.sum()
    return x_fused, P_fused, x_models, P_models, mu


def _pack(x_fused, P_fused, x_models, P_models, mu):
    """Assemble 26-D mean and 26×26 covariance from IMM components."""
    mean = np.zeros(_TOTAL)
    cov  = np.eye(_TOTAL) * 1e-8   # tiny diagonal baseline (non-singular)

    mean[_SL_FUSED] = x_fused
    cov[_SL_FUSED, _SL_FUSED] = P_fused

    for i, sl in enumerate(_SL_M):
        mean[sl]   = x_models[i]
        cov[sl, sl] = P_models[i]

    mean[_SL_PROB] = mu
    return mean, cov


# ══════════════════════════════════════════════════════════════════════════════
#  BehavioralIMMFilter  –  main public class
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralIMMFilter:
    """IMM filter that is a drop-in replacement for BehavioralEKFFilter.

    Same public interface:
        initiate(measurement)                              → (mean_26, cov_26x26)
        predict(mean, covariance, other_track_means=None)  → (mean_26, cov_26x26)
        update(mean, covariance, measurement)              → (mean_26, cov_26x26)
        gating_distance(mean, cov, measurements, ...)      → ndarray of distances
        project(mean, covariance)                          → (mean_4, cov_4x4)
    """

    def __init__(
        self,
        dt: float = 1.0,
        mu_init: tuple = (0.6, 0.3, 0.1),
        measurement_noise_std: float = 0.05,
        ca_decel: float = 0.85,
        social_force_threshold: float = 1.0,
    ):
        """
        Args:
            dt: Time step (frames; 1 = one detection cycle).
            mu_init: Initial mode probability vector (CV, CT, CA). Auto-normalised.
            measurement_noise_std: Std dev of detection noise in normalised image
                coords.  0.05 is more realistic than the 0.002 used in the
                original EKF (which over-trusted raw YOLOv9 boxes).
            ca_decel: Velocity damping factor γ for the CA model (0 < γ < 1).
                      0.85 ≈ 15% speed reduction per frame.
            social_force_threshold: Social-force magnitude (same units as pixel
                positions) above which we boost transition probability toward
                CT / CA.
        """
        self._dt  = dt
        self._ca_decel = ca_decel
        self._sf_thresh = social_force_threshold

        self._mu_init = np.array(mu_init, dtype=np.float64)
        self._mu_init /= self._mu_init.sum()

        # Measurement noise  R  (2×2) for position observations [px, py]
        self._R = np.diag([measurement_noise_std**2, measurement_noise_std**2])

        # ── Baseline Markov transition matrix Π (row i → col j) ──────────────
        # High self-transition probability; small cross-model leakage.
        self._Pi_base = np.array([
            [0.90, 0.07, 0.03],   # CV → (CV, CT, CA)
            [0.10, 0.85, 0.05],   # CT → (CV, CT, CA)
            [0.15, 0.05, 0.80],   # CA → (CV, CT, CA)
        ], dtype=np.float64)

        # ── Per-model process noise Q (5×5) ──────────────────────────────────
        # CV: tight on ω (keeps it near zero), moderate elsewhere
        self._Q = [
            np.diag([1.0, 1.0, 1.5, 2.0, 0.1])**2,   # CV
            np.diag([1.0, 1.0, 1.9, 3.0, 3.0])**2,   # CT  (same as behavioral_ekf_2)
            np.diag([1.5, 1.5, 5.0, 2.0, 0.1])**2,   # CA  (large on v for stop/go)
        ]

        # DeepSORT-style relative noise weights (kept identical to KalmanFilter)
        self._w_pos = 1.0 / 20
        self._w_vel = 1.0 / 160

    # ── Public API ────────────────────────────────────────────────────────────

    def initiate(self, measurement):
        """Create a new track from a 4-D detection [x, y, a, h].

        Returns 26-D mean and 26×26 covariance.
        """
        z  = np.asarray(measurement, dtype=np.float64)
        h  = z[3]

        # ── Fused 8-D initial state (zero velocity, same as KalmanFilter) ────
        x_fused = np.r_[z, np.zeros(4)]
        std_8d  = [
            2 * self._w_pos * h,  2 * self._w_pos * h,  1e-2,  2 * self._w_pos * h,
            10 * self._w_vel * h, 10 * self._w_vel * h, 1e-5, 10 * self._w_vel * h,
        ]
        P_fused = np.diag(np.square(std_8d))

        # ── Per-model 5-D initial states (position known, motion unknown) ─────
        x5_init = np.array([z[0], z[1], 0.0, 0.0, 0.0])
        P5_init = np.diag([
            (2  * self._w_pos * h) ** 2,
            (2  * self._w_pos * h) ** 2,
            (10 * self._w_vel * h) ** 2,
            0.1,    # heading φ  (unknown at initialisation)
            0.01,   # turn rate ω
        ])

        x_models = [x5_init.copy() for _ in range(_N)]
        P_models = [P5_init.copy() for _ in range(_N)]

        return _pack(x_fused, P_fused, x_models, P_models, self._mu_init.copy())

    def predict(self, mean, covariance, other_track_means=None):
        """IMM prediction:  Mixing → Mode-conditioned prediction → Fusion.

        IMM Interaction (Mixing) Step
        ─────────────────────────────
        Given mode probabilities μᵢ and transition matrix Π:

          Predicted mode probability:
              c̄ⱼ = Σᵢ Πᵢⱼ · μᵢ                         (1)

          Mixing weight (probability that model i was active
          given model j is active at this step):
              μᵢ|ⱼ = Πᵢⱼ · μᵢ / c̄ⱼ                     (2)

          Mixed initial condition for model j:
              x̂⁰ⱼ = Σᵢ μᵢ|ⱼ · x̂ᵢ                      (3)
              P⁰ⱼ  = Σᵢ μᵢ|ⱼ · [Pᵢ + (x̂ᵢ−x̂⁰ⱼ)(x̂ᵢ−x̂⁰ⱼ)ᵀ] (4)

        Args:
            mean: 26-D packed mean from previous step.
            covariance: 26×26 packed covariance from previous step.
            other_track_means: list of other tracks' 26-D means (for social force).

        Returns:
            (mean_pred, cov_pred): 26-D, 26×26
        """
        x_fused, P_fused, x_models, P_models, mu = _unpack(mean, covariance)
        dt = self._dt

        # ── Social force ──────────────────────────────────────────────────────
        # Compute inter-pedestrian repulsion using the [x, y] of each neighbour.
        sf    = np.zeros(2)
        sf_mag = 0.0
        if other_track_means is not None and len(other_track_means) > 0:
            other_xy = [m[:2] for m in other_track_means]  # take [x,y] from 8-D prefix
            x5_ref   = _mean_8_to_5(x_fused)
            sf       = _compute_social_force(x5_ref, other_xy)
            sf_mag   = float(np.linalg.norm(sf))

        # ── Dynamic Π: boost CT / CA when pedestrians are crowding ───────────
        # If social-force magnitude exceeds threshold, redistribute transition
        # probability mass away from CV toward CT (turning) and CA (stopping).
        Pi = self._Pi_base.copy()
        if sf_mag > self._sf_thresh:
            # delta saturates at 0.30 to prevent degenerate rows
            delta = min(0.30, 0.15 * sf_mag / self._sf_thresh)
            for i in range(_N):
                shift = delta * Pi[i, CV_IDX]
                Pi[i, CT_IDX] += 0.60 * shift
                Pi[i, CA_IDX] += 0.40 * shift
                Pi[i, CV_IDX] -= shift
            # Renormalise each row to sum to 1
            Pi = (Pi.T / Pi.sum(axis=1)).T

        # ── Step 1: Predicted mode probabilities  c̄ⱼ  (equation 1) ──────────
        c_bar = Pi.T @ mu                   # shape (N,)
        c_bar = np.maximum(c_bar, 1e-8)
        c_bar /= c_bar.sum()               # numerical safety renorm

        # ── Step 2: Mixing weights  μᵢ|ⱼ  (equation 2) ──────────────────────
        # mixing_w[i, j] = Πᵢⱼ · μᵢ / c̄ⱼ
        mixing_w = (Pi * mu[:, np.newaxis]) / c_bar[np.newaxis, :]  # (N, N)

        # ── Step 3: Mixed initial conditions for each model j  (eqs 3–4) ─────
        x_mix, P_mix = [], []
        for j in range(_N):
            # Mixed mean: weighted sum of per-model means
            xm = sum(mixing_w[i, j] * x_models[i] for i in range(_N))
            # Mixed covariance: include spread-of-means term
            Pm = np.zeros((5, 5))
            for i in range(_N):
                dx  = x_models[i] - xm
                Pm += mixing_w[i, j] * (P_models[i] + np.outer(dx, dx))
            x_mix.append(xm)
            P_mix.append(_nearest_psd(Pm))

        # ── Step 4: Mode-conditioned prediction ──────────────────────────────
        x_pred, P_pred = [], []

        xp, Pp = _cv_predict(x_mix[CV_IDX], P_mix[CV_IDX], self._Q[CV_IDX], dt, sf)
        x_pred.append(xp); P_pred.append(Pp)

        xp, Pp = _ct_predict(x_mix[CT_IDX], P_mix[CT_IDX], self._Q[CT_IDX], dt, sf)
        x_pred.append(xp); P_pred.append(Pp)

        xp, Pp = _ca_predict(x_mix[CA_IDX], P_mix[CA_IDX], self._Q[CA_IDX], dt, sf,
                              self._ca_decel)
        x_pred.append(xp); P_pred.append(Pp)

        # ── Step 5: Fuse predicted states with predicted mode probs c̄ⱼ ────────
        # x̂_fused = Σⱼ c̄ⱼ · x̂_pred_j
        # P_fused  = Σⱼ c̄ⱼ · [P_pred_j + (x̂_pred_j − x̂_fused)(…)ᵀ]
        x5_fused = sum(c_bar[j] * x_pred[j] for j in range(_N))
        P5_fused = np.zeros((5, 5))
        for j in range(_N):
            dx = x_pred[j] - x5_fused
            P5_fused += c_bar[j] * (P_pred[j] + np.outer(dx, dx))
        P5_fused = _nearest_psd(P5_fused)

        # Convert fused 5-D → 8-D DeepSORT state
        x_fused_new, P_fused_new = _mean_cov_5_to_8(
            x5_fused, P5_fused, x_fused, P_fused, dt, self._w_pos, self._w_vel
        )

        return _pack(x_fused_new, _nearest_psd(P_fused_new), x_pred, P_pred, c_bar)

    def update(self, mean, covariance, measurement):
        """IMM update:  Mode-conditioned update → Mode prob update → Fusion.

        Mode Probability Update
        ───────────────────────
        Each model j produces a likelihood Λⱼ from its innovation:
            Λⱼ = N(zₖ − Hx̂ⱼ; 0, Sⱼ)                   (5)

        Updated mode probability:
            μⱼ(k) ∝ Λⱼ · c̄ⱼ                            (6)

        Fused estimate:
            x̂(k|k)  = Σⱼ μⱼ · x̂ⱼ(k|k)                (7)
            P(k|k)   = Σⱼ μⱼ · [Pⱼ + (x̂ⱼ − x̂)(x̂ⱼ − x̂)ᵀ] (8)

        Args:
            mean: 26-D packed mean (output of predict).
            covariance: 26×26 packed covariance.
            measurement: 4-D observation [x, y, a, h].

        Returns:
            (mean_upd, cov_upd): 26-D, 26×26
        """
        x_fused, P_fused, x_pred, P_pred, mu = _unpack(mean, covariance)
        z    = np.asarray(measurement, dtype=np.float64)
        z_xy = z[:2]   # only position is observed by the 5-D EKF

        # ── Step 6: Mode-conditioned update (equation 5) ─────────────────────
        x_upd, P_upd, log_liks = [], [], []
        for j in range(_N):
            xu, Pu, innov, S = _ekf_update_5d(x_pred[j], P_pred[j], z_xy, self._R)
            x_upd.append(xu)
            P_upd.append(Pu)
            log_liks.append(_log_gaussian_likelihood(innov, S))

        # ── Step 7: Mode probability update (equation 6) ─────────────────────
        # Log-sum-exp trick: shift by max to avoid underflow before exponentiating.
        log_liks = np.array(log_liks)
        log_liks -= log_liks.max()
        likelihoods = np.exp(log_liks)

        mu_new = likelihoods * mu        # Λⱼ · c̄ⱼ  (c̄ is stored as mu after predict)
        mu_new = np.maximum(mu_new, 1e-8)
        mu_new /= mu_new.sum()

        # ── Step 8: Fused updated estimate (equations 7–8) ───────────────────
        x5_fused_upd = sum(mu_new[j] * x_upd[j] for j in range(_N))
        P5_fused_upd = np.zeros((5, 5))
        for j in range(_N):
            dx = x_upd[j] - x5_fused_upd
            P5_fused_upd += mu_new[j] * (P_upd[j] + np.outer(dx, dx))
        P5_fused_upd = _nearest_psd(P5_fused_upd)

        # Rebuild 8-D state from fused 5-D
        v, phi = x5_fused_upd[2], x5_fused_upd[3]
        vx, vy = _xy_from_vphi(v, phi)
        a_new, h_new = z[2], z[3]
        x_fused_new = np.array([
            x5_fused_upd[0], x5_fused_upd[1],
            a_new, h_new, vx, vy,
            x_fused[6], x_fused[7],    # carry-forward va, vh
        ])

        # Build 8-D covariance from 5-D (same Jacobian as behavioral_ekf_2.py)
        J_vy = np.array([
            [np.cos(phi), -v * np.sin(phi)],
            [np.sin(phi),  v * np.cos(phi)],
        ])
        P_fused_new = np.eye(8)
        P_fused_new[0:2, 0:2] = P5_fused_upd[0:2, 0:2]
        P_fused_new[4:6, 4:6] = J_vy @ P5_fused_upd[2:4, 2:4] @ J_vy.T
        P_fused_new[0:2, 4:6] = P5_fused_upd[0:2, 2:4] @ J_vy.T
        P_fused_new[4:6, 0:2] = P_fused_new[0:2, 4:6].T
        P_fused_new[2:4, 2:4] = np.diag(np.square([1e-2, self._w_pos * h_new]))
        P_fused_new[6:8, 6:8] = np.diag(np.square([1e-5, self._w_vel * h_new]))

        return _pack(x_fused_new, _nearest_psd(P_fused_new), x_upd, P_upd, mu_new)

    def project(self, mean, covariance):
        """Project to 4-D measurement space [x, y, a, h].

        Used by gating_distance and (indirectly) by the tracker for gate checking.
        """
        x_f = mean[_SL_FUSED]
        P_f = covariance[_SL_FUSED, _SL_FUSED]
        h   = x_f[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-1, self._w_pos * h]
        innov_cov = np.diag(np.square(std))
        mean_proj = x_f[:4].copy()
        cov_proj  = _nearest_psd(P_f[:4, :4].copy() + innov_cov)
        return mean_proj, cov_proj

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Squared Mahalanobis distance between this track and a set of detections.

        Returns array of length len(measurements) for use in tracker._match().
        """
        mean_proj, cov_proj = self.project(mean, covariance)
        meas = np.asarray(measurements)

        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj  = cov_proj[:2, :2]
            meas      = meas[:, :2]

        # Trace-relative regularisation: scales with the matrix magnitude so it
        # remains effective even when covariance has been heavily inflated by
        # track.py's occlusion multiplier.  Fixed epsilons (e.g. 1e-4) become
        # negligible once covariance values reach 1e6+.
        n = len(cov_proj)
        reg = max(np.trace(cov_proj) * 1e-6, 1e-4)
        cov_proj = cov_proj + np.eye(n) * reg

        try:
            chol = np.linalg.cholesky(cov_proj)
            d = meas - mean_proj
            z = scipy.linalg.solve_triangular(
                chol, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            # Truly degenerate after regularisation — return large distances so
            # this track is skipped in association rather than crashing.
            return np.full(meas.shape[0], 1e5)