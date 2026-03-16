"""
behavioral_imm.py  –  Interacting Multiple Model (IMM) filter for pedestrian tracking.
                       Version 3: Pure 3D world-frame tracking.

Blends three kinematic sub-filters:
  Model 0 (CV) : Constant Velocity  – steady walking, no turn
  Model 1 (CT) : Coordinated Turn   – turning / swerving pedestrian
  Model 2 (CA) : Constant Accel/Stop – yielding or stopping

Internal EKF state per model (7-D):  [pX, pZ, v, phi, omega, pY, vY]
  pX    – camera-right position  (metres)
  pZ    – camera-forward / depth (metres)
  v     – horizontal speed in XZ plane
  phi   – heading angle in XZ plane (radians)
  omega – turn rate in XZ plane (rad/frame)
  pY    – camera-down position   (metres)
  vY    – vertical velocity      (metres/frame)

Measurements: 3-D world positions [pX, pZ, pY] from depth unprojection.
Gating / association: Mahalanobis distance in 3-D world space [pX, pZ, pY].

Packed state layout (24-D mean, 24×24 covariance):
  mean[ 0: 7]  – CV model 7-D state
  mean[ 7:14]  – CT model 7-D state
  mean[14:21]  – CA model 7-D state
  mean[21:24]  – mode probabilities  [mu_cv, mu_ct, mu_ca]

  cov[ 0: 7,  0: 7]  – CV  7x7 covariance
  cov[ 7:14,  7:14]  – CT  7x7 covariance
  cov[14:21, 14:21]  – CA  7x7 covariance

NOTE: track.py inflates covariance by up to 4x during occlusions.  This is
compatible with this filter; the cap prevents numerical blowup after long gaps.
"""

import warnings
import numpy as np
import scipy.linalg

from . import kalman_filter as _kf

# Suppress scipy's LinAlgWarning for ill-conditioned matrices — handled explicitly
# via trace-relative regularisation throughout this module.
warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)

chi2inv95 = _kf.chi2inv95  # re-export for tracker.py gating

# ── Model index aliases ────────────────────────────────────────────────────────
CV_IDX, CT_IDX, CA_IDX = 0, 1, 2
_N = 3  # number of models

# ── State packing dimensions ───────────────────────────────────────────────────
_DIM_INNER = 7
_DIM_PROB  = _N
_TOTAL     = _N * _DIM_INNER + _DIM_PROB  # 24

_SL_M    = [slice(i * 7, (i + 1) * 7) for i in range(_N)]  # [0:7, 7:14, 14:21]
_SL_PROB = slice(21, 24)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════════════════════


def _nearest_psd(M: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite matrix
    via eigenvalue clamping (Higham 1988).

    After reconstruction via V @ diag(lambda) @ V^T, floating-point rounding can
    leave eigenvalues at ~-1e-16.  Adding min_eig*I guarantees strict PD so
    that np.linalg.cholesky always succeeds.
    """
    B = (M + M.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, min_eig)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    result += np.eye(len(result)) * min_eig
    return result


def _log_gaussian_likelihood(innov: np.ndarray, S: np.ndarray) -> float:
    """Log-likelihood of innovation under N(0, S).  Returns -inf on failure."""
    n = innov.shape[0]
    try:
        chol = np.linalg.cholesky(S)
        z = scipy.linalg.solve_triangular(chol, innov, lower=True, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (n * np.log(2.0 * np.pi) + log_det + float(z @ z))
    except np.linalg.LinAlgError:
        return -1e10


# ══════════════════════════════════════════════════════════════════════════════
#  Per-model state transition functions  (operate on 7-D internal state)
#
#  State layout:  [pX, pZ, v, phi, omega, pY, vY]
#    pX, pZ = horizontal position in camera XZ plane (right, forward)
#    v      = horizontal speed
#    phi    = heading angle in XZ plane
#    omega  = turn rate
#    pY     = camera-down position (vertical)
#    vY     = vertical velocity
# ══════════════════════════════════════════════════════════════════════════════

def _cv_predict(x7: np.ndarray, P7: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray):
    """Constant Velocity prediction (omega forced to 0).

    Social force injected as kinematic input in XZ plane:
        pX' += 0.5*sf[0]*dt^2,  pZ' += 0.5*sf[1]*dt^2
    """
    pX, pZ, v, phi, _omega, pY, vY = x7

    x_pred = np.array([
        pX + v * np.cos(phi) * dt + 0.5 * sf[0] * dt**2,
        pZ + v * np.sin(phi) * dt + 0.5 * sf[1] * dt**2,
        v,
        phi,
        0.0,          # CV clamps omega to zero
        pY + vY * dt,
        vY,
    ])

    # Jacobian F (7x7)
    F = np.eye(7)
    F[0, 2] =  np.cos(phi) * dt
    F[0, 3] = -v * np.sin(phi) * dt
    F[1, 2] =  np.sin(phi) * dt
    F[1, 3] =  v * np.cos(phi) * dt
    F[5, 6] =  dt   # pY += vY*dt

    P_pred = F @ P7 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


def _ct_predict(x7: np.ndarray, P7: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray):
    """Coordinated Turn prediction in XZ plane.

    Straight-line limit when |omega| < 1e-4.
    Full Jacobian includes d/d_omega column (critical for covariance accuracy
    during curved motion — missing in original behavioral_ekf.py).
    Vertical state (pY, vY) propagates as constant velocity.
    """
    pX, pZ, v, phi, omega, pY, vY = x7

    x_pred = x7.copy()
    if abs(omega) < 1e-4:
        x_pred[0] = pX + v * np.cos(phi) * dt
        x_pred[1] = pZ + v * np.sin(phi) * dt
    else:
        x_pred[0] = pX + (v / omega) * (np.sin(phi + omega * dt) - np.sin(phi))
        x_pred[1] = pZ + (v / omega) * (np.cos(phi) - np.cos(phi + omega * dt))
    x_pred[3] = phi + omega * dt
    # Social force in XZ plane
    x_pred[0] += 0.5 * sf[0] * dt**2
    x_pred[1] += 0.5 * sf[1] * dt**2
    # Vertical constant-velocity propagation
    x_pred[5] = pY + vY * dt
    x_pred[6] = vY

    # Full Jacobian (7x7)
    F = np.eye(7)
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
        # d/d_omega column (essential for CT covariance accuracy)
        F[0, 4] = v * (omega * dt * c_phi_odt - s_phi_odt + s_phi) / omega**2
        F[1, 4] = v * (omega * dt * s_phi_odt + c_phi_odt - c_phi) / omega**2
    F[3, 4] = dt   # d_phi / d_omega
    F[5, 6] = dt   # pY += vY*dt

    P_pred = F @ P7 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


def _ca_predict(x7: np.ndarray, P7: np.ndarray, Q: np.ndarray, dt: float,
                sf: np.ndarray, gamma: float = 0.85, gamma_y: float = 0.9):
    """Constant Deceleration / Stop prediction.

    Speed and vertical velocity are damped each frame:
        v'  = gamma * v
        vY' = gamma_y * vY
    """
    pX, pZ, v, phi, _omega, pY, vY = x7

    x_pred = np.array([
        pX + v * np.cos(phi) * dt + 0.5 * sf[0] * dt**2,
        pZ + v * np.sin(phi) * dt + 0.5 * sf[1] * dt**2,
        gamma * v,
        phi,
        0.0,
        pY + vY * dt,
        gamma_y * vY,
    ])

    F = np.eye(7)
    F[0, 2] =  np.cos(phi) * dt
    F[0, 3] = -v * np.sin(phi) * dt
    F[1, 2] =  np.sin(phi) * dt
    F[1, 3] =  v * np.cos(phi) * dt
    F[2, 2] =  gamma      # velocity damping in Jacobian
    F[5, 6] =  dt
    F[6, 6] =  gamma_y    # vertical velocity damping

    P_pred = F @ P7 @ F.T + Q
    return x_pred, _nearest_psd(P_pred)


# ══════════════════════════════════════════════════════════════════════════════
#  EKF measurement update  (7-D state, 3-D world measurement)
# ══════════════════════════════════════════════════════════════════════════════

def _ekf_update_7d_3d(x7: np.ndarray, P7: np.ndarray,
                      z_world: np.ndarray, R3: np.ndarray):
    """3D EKF update.  Measurement z_world = [pX_meas, pZ_meas, pY_meas] (metres).

    Observation matrix H (3x7): observes pX (idx 0), pZ (idx 1), pY (idx 5).
    Joseph-form covariance update for numerical stability.
    Returns (x_new, P_new, innovation, innovation_cov).
    """
    H = np.zeros((3, 7))
    H[0, 0] = 1.0   # observe pX
    H[1, 1] = 1.0   # observe pZ
    H[2, 5] = 1.0   # observe pY

    innov = z_world - H @ x7
    S     = (H @ P7 @ H.T + R3)
    S     = (S + S.T) * 0.5

    reg_S = max(np.trace(S) * 1e-6, 1e-8)
    S_reg = S + np.eye(3) * reg_S

    try:
        K = scipy.linalg.solve(S_reg, H @ P7, assume_a='pos').T
    except np.linalg.LinAlgError:
        return x7.copy(), P7.copy(), innov, S_reg

    x_new = x7 + K @ innov
    I_KH  = np.eye(7) - K @ H
    P_new = I_KH @ P7 @ I_KH.T + K @ R3 @ K.T

    return x_new, _nearest_psd(P_new), innov, S_reg


# ══════════════════════════════════════════════════════════════════════════════
#  Social force  (in XZ world plane)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_social_force(x7: np.ndarray, other_xz: list,
                          A: float = 2.0, B: float = 0.5,
                          r: float = 0.3, max_dist: float = 3.0) -> np.ndarray:
    """Exponential repulsion from neighbouring pedestrians.

    f = A * exp((2r - d) / B) * n_ij  for d < max_dist

    Args:
        other_xz: list of [pX, pZ] for each neighbour track.
        A, B, r: standard social force model parameters.
        max_dist: ignore neighbours beyond this distance (metres).
    Returns:
        2-D force vector [fX, fZ].
    """
    f = np.zeros(2)
    pX, pZ = x7[0], x7[1]
    for nx, nz in other_xz:
        d = np.hypot(nx - pX, nz - pZ)
        if d < 0.01 or d >= max_dist:
            continue
        n_ij = np.array([pX - nx, pZ - nz]) / d
        f += A * np.exp((2.0 * r - d) / B) * n_ij
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  State pack / unpack
# ══════════════════════════════════════════════════════════════════════════════

def _unpack(mean: np.ndarray, cov: np.ndarray):
    """Decompose 24-D mean / 24x24 covariance into IMM components."""
    x_models = [mean[sl].copy() for sl in _SL_M]
    P_models = [cov[sl, sl].copy() for sl in _SL_M]
    mu       = mean[_SL_PROB].copy()
    mu       = np.maximum(mu, 1e-8)
    mu      /= mu.sum()
    return x_models, P_models, mu


def _pack(x_models, P_models, mu):
    """Assemble 24-D mean and 24x24 covariance from IMM components."""
    mean = np.zeros(_TOTAL)
    cov  = np.eye(_TOTAL) * 1e-8   # tiny diagonal baseline

    for i, sl in enumerate(_SL_M):
        mean[sl]    = x_models[i]
        cov[sl, sl] = P_models[i]

    mean[_SL_PROB] = mu
    return mean, cov


# ══════════════════════════════════════════════════════════════════════════════
#  BehavioralIMMFilter  –  main public class
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralIMMFilter:
    """IMM filter operating purely in 3-D world-frame coordinates.

    Requires depth-unprojected 3-D world positions for both initiate and update.
    Gating / association uses Mahalanobis distance in [pX, pZ, pY] world space.

    Public interface:
        initiate(world_pos)                                -> (mean_24, cov_24x24)
        predict(mean, covariance, other_track_means=None) -> (mean_24, cov_24x24)
        update(mean, covariance, world_pos)               -> (mean_24, cov_24x24)
        gating_distance(mean, cov, world_positions)       -> ndarray of distances
    """

    def __init__(
        self,
        dt: float = 1.0,
        mu_init: tuple = (0.6, 0.3, 0.1),
        ca_decel: float = 0.85,
        ca_decel_y: float = 0.9,
    ):
        self._dt         = dt
        self._ca_decel   = ca_decel
        self._ca_decel_y = ca_decel_y

        self._mu_init = np.array(mu_init, dtype=np.float64)
        self._mu_init /= self._mu_init.sum()

        # ── Measurement noise (3-D, metres) ───────────────────────────────────
        # sigma_XZ = 0.05 m  (lateral + depth from depth sensor)
        # sigma_Y  = 0.10 m  (vertical; depth noisier for small objects)
        self._R3 = np.diag([0.05**2, 0.05**2, 0.10**2])

        # ── Markov transition matrix Pi (row i -> col j) ──────────────────────
        self._Pi_base = np.array([
            [0.90, 0.07, 0.03],   # CV → (CV, CT, CA)
            [0.10, 0.85, 0.05],   # CT → (CV, CT, CA)
            [0.15, 0.05, 0.80],   # CA → (CV, CT, CA)
        ], dtype=np.float64)

        # ── Social force parameters (world frame, metres) ─────────────────────
        self._sf_max_dist  = 3.0    # metres
        self._sf_threshold = 0.5    # m/s^2

        # ── Per-model process noise Q (7x7, world-frame metres) ───────────────
        # Order: [pX, pZ, v, phi, omega, pY, vY]
        # Pedestrian ~1.5 m/s, dt=1/30 s => ~0.05 m/frame
        self._Q = [
            np.diag([1e-1, 1e-1, 0.03, 0.01, 1e-1, 5e-2, 5e-2])**2,   # CV
            np.diag([1e-2, 1e-2, 0.02, 0.010, 0.05, 5e-3, 5e-4])**2,   # CT
            np.diag([1e-2, 1e-2, 0.10, 0.005, 1e-5, 5e-3, 5e-3])**2,   # CA
        ]

    # ── Public API ─────────────────────────────────────────────────────────────

    def initiate(self, world_pos):
        """Create a new track from a 3-D world position.

        Args:
            world_pos: 3-D array [pX, pZ, pY] in camera frame (metres).

        Returns:
            (mean_24, cov_24x24) packed state.
        """
        pX, pZ, pY = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])

        # Position known from depth; velocity unknown at first detection
        x7_init = np.array([pX, pZ, 0.0, 0.0, 0.0, pY, 0.0])

        P7_init = np.diag([
            0.05**2,    # pX  — depth sensor lateral accuracy
            0.05**2,    # pZ  — depth sensor forward accuracy
            1.0**2,     # v   — unknown initially
            0.5**2,     # phi — unknown initially
            0.1**2,     # omega
            0.10**2,    # pY  — depth sensor vertical accuracy
            0.5**2,     # vY  — unknown initially
        ])

        x_models = [x7_init.copy() for _ in range(_N)]
        P_models = [P7_init.copy() for _ in range(_N)]

        return _pack(x_models, P_models, self._mu_init.copy())

    def initiate_no_depth(self):
        """Placeholder state when no world_pos is available.
        Sets position to zero with very large uncertainty so the filter
        contributes nothing until a real depth measurement arrives.
        """
        x7 = np.zeros(7)
        P7 = np.eye(7) * 1e6
        x_models = [x7.copy() for _ in range(_N)]
        P_models = [P7.copy() for _ in range(_N)]
        return _pack(x_models, P_models, self._mu_init.copy())

    def predict(self, mean, covariance, other_track_means=None):
        """IMM prediction: Mixing -> Mode-conditioned prediction -> Fusion.

        Args:
            mean:              24-D packed mean from previous step.
            covariance:        24x24 packed covariance.
            other_track_means: list of other tracks' 24-D means (for social force).

        Returns:
            (mean_pred, cov_pred): 24-D, 24x24.
        """
        x_models, P_models, mu = _unpack(mean, covariance)
        dt = self._dt

        # ── Social force (world XZ plane, metres) ─────────────────────────────
        sf    = np.zeros(2)
        sf_mag = 0.0
        if other_track_means is not None and len(other_track_means) > 0:
            other_xz = [[float(m[_SL_M[CV_IDX].start]),
                         float(m[_SL_M[CV_IDX].start + 1])]
                        for m in other_track_means]
            sf    = _compute_social_force(x_models[CV_IDX], other_xz,
                                          max_dist=self._sf_max_dist)
            sf_mag = float(np.linalg.norm(sf))

        # ── Dynamic Pi: boost CT / CA when crowding is detected ───────────────
        Pi = self._Pi_base.copy()
        if sf_mag > self._sf_threshold:
            delta = min(0.30, 0.15 * sf_mag / self._sf_threshold)
            for i in range(_N):
                shift = delta * Pi[i, CV_IDX]
                Pi[i, CT_IDX] += 0.60 * shift
                Pi[i, CA_IDX] += 0.40 * shift
                Pi[i, CV_IDX] -= shift
            Pi = (Pi.T / Pi.sum(axis=1)).T

        # ── Step 1: Predicted mode probabilities c̄ⱼ ──────────────────────────
        c_bar = Pi.T @ mu
        c_bar = np.maximum(c_bar, 1e-8)
        c_bar /= c_bar.sum()

        # ── Step 2: Mixing weights mu_{i|j} ──────────────────────────────────
        mixing_w = (Pi * mu[:, np.newaxis]) / c_bar[np.newaxis, :]  # (N, N)

        # ── Step 3: Mixed initial conditions for each model j ─────────────────
        x_mix, P_mix = [], []
        for j in range(_N):
            xm = sum(mixing_w[i, j] * x_models[i] for i in range(_N))
            Pm = np.zeros((7, 7))
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
                              self._ca_decel, self._ca_decel_y)
        x_pred.append(xp); P_pred.append(Pp)

        return _pack(x_pred, P_pred, c_bar)

    def update(self, mean, covariance, world_pos):
        """IMM update: Mode-conditioned update -> Mode prob update -> Fusion.

        Args:
            mean:       24-D packed mean (output of predict).
            covariance: 24x24 packed covariance.
            world_pos:  3-D [pX, pZ, pY] in camera frame (metres).

        Returns:
            (mean_upd, cov_upd): 24-D, 24x24.
        """
        x_models, P_models, mu = _unpack(mean, covariance)
        z_world = np.asarray(world_pos, dtype=np.float64)  # [pX, pZ, pY]

        # ── Mode-conditioned update ───────────────────────────────────────────
        x_upd, P_upd, log_liks = [], [], []
        for j in range(_N):
            xu, Pu, innov, S = _ekf_update_7d_3d(
                x_models[j], P_models[j], z_world, self._R3)
            x_upd.append(xu)
            P_upd.append(Pu)
            log_liks.append(_log_gaussian_likelihood(innov, S))

        # ── Mode probability update ───────────────────────────────────────────
        log_liks = np.array(log_liks)
        log_liks -= log_liks.max()
        likelihoods = np.exp(log_liks)

        mu_new = likelihoods * mu
        mu_new = np.maximum(mu_new, 1e-8)
        mu_new /= mu_new.sum()

        return _pack(x_upd, P_upd, mu_new)

    def gating_distance(self, mean, covariance, world_positions):
        """Squared Mahalanobis distance in 3-D world space [pX, pZ, pY].

        Args:
            mean:            24-D packed mean.
            covariance:      24x24 packed covariance.
            world_positions: array of shape (N, 3) — one [pX, pZ, pY] per detection.

        Returns:
            Array of length N with squared Mahalanobis distances.
        """
        x_models, P_models, mu = _unpack(mean, covariance)

        # Fused 7-D world state: probability-weighted mean and covariance
        x7_fused = sum(mu[j] * x_models[j] for j in range(_N))
        P7_fused = np.zeros((7, 7))
        for j in range(_N):
            dx = x_models[j] - x7_fused
            P7_fused += mu[j] * (P_models[j] + np.outer(dx, dx))
        P7_fused = _nearest_psd(P7_fused)

        # Extract [pX, pZ, pY] sub-state (indices 0, 1, 5 of 7-D state)
        idx_3d  = [0, 1, 5]
        mean_3d = x7_fused[idx_3d]
        cov_3d  = _nearest_psd(P7_fused[np.ix_(idx_3d, idx_3d)] + self._R3)

        reg = max(np.trace(cov_3d) * 1e-6, 1e-4)
        cov_3d = cov_3d + np.eye(3) * reg

        meas = np.asarray(world_positions)
        if meas.ndim == 1:
            meas = meas[np.newaxis, :]

        try:
            chol = np.linalg.cholesky(cov_3d)
            d = meas - mean_3d
            z = scipy.linalg.solve_triangular(
                chol, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            return np.full(meas.shape[0], 1e5)
