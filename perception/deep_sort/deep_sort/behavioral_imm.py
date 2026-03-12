"""
behavioral_imm.py  –  Interacting Multiple Model (IMM) filter for pedestrian tracking.
                       Version 2: 3D world-frame tracking with depth images.

Drop-in replacement for BehavioralEKFFilter (behavioral_ekf_2.py).

Blends three kinematic sub-filters:
  Model 0 (CV) : Constant Velocity  – steady walking, no turn
  Model 1 (CT) : Coordinated Turn   – turning / swerving pedestrian
  Model 2 (CA) : Constant Accel/Stop – yielding or stopping

Internal EKF state per model (7-D):  [pX, pZ, v, phi, omega, pY, vY]
  pX    – camera-right position  (metres in 3D mode; pixel-x in 2D fallback)
  pZ    – camera-forward / depth (metres in 3D mode; pixel-y in 2D fallback)
  v     – horizontal speed in XZ plane
  phi   – heading angle in XZ plane (radians)
  omega – turn rate in XZ plane (rad/frame)
  pY    – camera-down position   (metres in 3D mode; 0 in 2D fallback)
  vY    – vertical velocity      (metres/frame in 3D mode; 0 in 2D fallback)

External DeepSORT state (8-D):  [x, y, a, h, vx, vy, va, vh]

Packed state layout (32-D mean, 32×32 covariance):
  mean[ 0: 8]  – fused DeepSORT state
  mean[ 8:15]  – CV model 7-D state
  mean[15:22]  – CT model 7-D state
  mean[22:29]  – CA model 7-D state
  mean[29:32]  – mode probabilities  [mu_cv, mu_ct, mu_ca]

  cov[ 0:8,   0:8]   – fused 8x8 covariance
  cov[ 8:15,  8:15]  – CV  7x7 covariance
  cov[15:22, 15:22]  – CT  7x7 covariance
  cov[22:29, 22:29]  – CA  7x7 covariance

When depth images are provided (world_pos is not None in initiate/update):
  - All three sub-models run in 3D camera-frame coordinates (metres).
  - Measurements are 3D positions [pX, pZ, pY] from depth unprojection.
  - Predicted 3D state is projected back to image-plane [x, y, ...] for the
    fused 8-D state that feeds gating / ReID association.

When no depth images are provided (world_pos is None):
  - Falls back to 2D image-frame tracking (same behaviour as the previous 5-D IMM).
  - pX = image x, pZ = image y, pY/vY = 0 (unused).

NOTE: track.py inflates covariance by up to 4x during occlusions.  This is
compatible with both modes; the cap prevents numerical blowup after long gaps.
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
_DIM_FUSED = 8
_DIM_INNER = 7
_DIM_PROB  = _N
_TOTAL     = _DIM_FUSED + _N * _DIM_INNER + _DIM_PROB  # 32

_SL_FUSED = slice(0, 8)
_SL_M     = [slice(8 + i * 7, 8 + (i + 1) * 7) for i in range(_N)]  # [8:15, 15:22, 22:29]
_SL_PROB  = slice(29, 32)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _vphi_from_xz(vx: float, vz: float):
    """Convert (vX, vZ) velocity to speed and heading angle in XZ plane."""
    v = np.hypot(vx, vz)
    phi = float(np.arctan2(vz, vx)) if v > 1e-8 else 0.0
    return v, phi


def _xz_from_vphi(v: float, phi: float):
    return v * np.cos(phi), v * np.sin(phi)


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
#  EKF measurement update  (7-D state)
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


def _ekf_update_7d_2d(x7: np.ndarray, P7: np.ndarray,
                      z_xy: np.ndarray, R2: np.ndarray):
    """2D fallback EKF update.  Measurement z_xy = [pX_pix, pZ_pix].

    Observation matrix H (2x7): observes pX (idx 0), pZ (idx 1).
    Used when no depth image is available.
    """
    H = np.zeros((2, 7))
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    innov = z_xy - H @ x7
    S     = (H @ P7 @ H.T + R2)
    S     = (S + S.T) * 0.5

    reg_S = max(np.trace(S) * 1e-6, 1e-8)
    S_reg = S + np.eye(2) * reg_S

    try:
        K = scipy.linalg.solve(S_reg, H @ P7, assume_a='pos').T
    except np.linalg.LinAlgError:
        return x7.copy(), P7.copy(), innov, S_reg

    x_new = x7 + K @ innov
    I_KH  = np.eye(7) - K @ H
    P_new = I_KH @ P7 @ I_KH.T + K @ R2 @ K.T

    return x_new, _nearest_psd(P_new), innov, S_reg


# ══════════════════════════════════════════════════════════════════════════════
#  7-D world state <-> 8-D DeepSORT state bridging
# ══════════════════════════════════════════════════════════════════════════════

def _mean_8_to_7(m8: np.ndarray, world_pos=None) -> np.ndarray:
    """Convert DeepSORT 8-D state to 7-D internal state.

    If world_pos = [pX, pZ, pY] is given (3D mode), use it directly for position.
    Otherwise, derive from pixel coords (2D fallback).
    """
    if world_pos is not None:
        pX, pZ, pY = world_pos
        # Velocity initial guess: derive from image-plane velocity is unreliable
        # at initiation — set to zero.
        return np.array([pX, pZ, 0.0, 0.0, 0.0, pY, 0.0])
    else:
        # 2D fallback: treat image (x, y) as (pX, pZ), pY=vY=0
        vx, vz = m8[4], m8[5]
        v, phi = _vphi_from_xz(vx, vz)
        return np.array([m8[0], m8[1], v, phi, 0.0, 0.0, 0.0])


def _cov_8_to_7(m8: np.ndarray, c8: np.ndarray) -> np.ndarray:
    """Project 8x8 DeepSORT covariance to 7x7 internal covariance (2D fallback).

    Maps (x, y) pos uncertainty and (vx, vz) velocity uncertainty.
    pY and vY components are initialised to small values.
    """
    vx, vz = m8[4], m8[5]
    v, phi = _vphi_from_xz(vx, vz)

    c7 = np.eye(7) * 1e-4

    # Position block [pX, pZ] from DeepSORT [x, y]
    c7[0:2, 0:2] = c8[0:2, 0:2]

    # Velocity block [v, phi] from [vx, vz] via Jacobian
    J_inv = np.array([
        [ np.cos(phi),                np.sin(phi)             ],
        [-np.sin(phi) / (v + 1e-8),   np.cos(phi) / (v + 1e-8)],
    ])
    c7[2:4, 2:4] = J_inv @ c8[4:6, 4:6] @ J_inv.T
    c7[0:2, 2:4] = c8[0:2, 4:6] @ J_inv.T
    c7[2:4, 0:2] = c7[0:2, 2:4].T

    # pY, vY — large initial uncertainty
    c7[5, 5] = 1e-2
    c7[6, 6] = 1e-4

    return _nearest_psd(c7)


def _bridge_7d_to_8d_3d(x7: np.ndarray, P7: np.ndarray,
                         m8_prev: np.ndarray, c8_prev: np.ndarray,
                         dt: float, w_pos: float, w_vel: float,
                         fx: float, fy: float, cx: float, cy: float):
    """Convert 7-D world-frame state to DeepSORT 8-D image-plane state (3D mode).

    Projects world position and velocity through the pinhole camera model.
    Propagates covariance through the projection Jacobian.
    Aspect ratio a and height h are carried forward with CV dynamics.
    """
    pX, pZ, v, phi, omega, pY, vY = x7
    pZ_safe = max(float(pZ), 0.1)   # avoid division by zero

    # World-frame horizontal velocities in XZ plane
    vX_w = v * np.cos(phi)
    vZ_w = v * np.sin(phi)

    # Image-plane position via pinhole projection
    u     = fx * pX / pZ_safe + cx
    v_img = fy * pY / pZ_safe + cy

    # Image-plane velocity (d/dt of projection)
    #   du/dt   = fx * (vX_w * pZ - pX * vZ_w) / pZ^2
    #   dv_img/dt = fy * (vY   * pZ - pY * vZ_w) / pZ^2
    vx_img = fx * (vX_w * pZ_safe - pX * vZ_w)  / pZ_safe**2
    vy_img = fy * (vY   * pZ_safe - pY * vZ_w)  / pZ_safe**2

    # Aspect ratio and height: constant-velocity propagation
    a  = m8_prev[2] + m8_prev[6] * dt
    h  = m8_prev[3] + m8_prev[7] * dt
    va, vh = m8_prev[6], m8_prev[7]

    m8_new = np.array([u, v_img, a, h, vx_img, vy_img, va, vh])

    # Covariance propagation: Jacobian J (4x7)
    # Rows: [du, dv_img, dvx_img, dvy_img]; Cols: [pX, pZ, v, phi, omega, pY, vY]
    J = np.zeros((4, 7))

    # d(u) / d(pX, pZ)
    J[0, 0] =  fx / pZ_safe
    J[0, 1] = -fx * pX / pZ_safe**2

    # d(v_img) / d(pY, pZ)
    J[1, 5] =  fy / pZ_safe
    J[1, 1] = -fy * pY / pZ_safe**2

    # d(vx_img) / d(pX, pZ, v, phi)
    J[2, 0] = -fx * vZ_w / pZ_safe**2
    J[2, 1] =  fx * (2.0 * pX * vZ_w - vX_w * pZ_safe) / pZ_safe**3
    J[2, 2] =  fx * (np.cos(phi) * pZ_safe - pX * np.sin(phi)) / pZ_safe**2
    J[2, 3] =  fx * v * (-np.sin(phi) * pZ_safe - pX * np.cos(phi)) / pZ_safe**2

    # d(vy_img) / d(pZ, v, phi, pY, vY)
    J[3, 1] =  fy * (2.0 * pY * vZ_w - vY * pZ_safe) / pZ_safe**3
    J[3, 2] = -fy * pY * np.sin(phi) / pZ_safe**2
    J[3, 3] = -fy * pY * v * np.cos(phi) / pZ_safe**2
    J[3, 5] = -fy * vZ_w / pZ_safe**2
    J[3, 6] =  fy / pZ_safe

    P_proj_4 = _nearest_psd(J @ P7 @ J.T)

    # Propagate (a, h, va, vh) sub-filter independently
    idx_ah = [2, 3, 6, 7]
    F_ah = np.eye(4); F_ah[0, 2] = dt; F_ah[1, 3] = dt
    std_ah = [w_pos * h, w_pos * h, 1e-5, w_vel * h]
    Q_ah = np.diag(np.square(std_ah))
    cov_ah_4 = F_ah @ c8_prev[np.ix_(idx_ah, idx_ah)] @ F_ah.T + Q_ah

    c8_new = np.eye(8) * 1e-8
    c8_new[0:2, 0:2] = P_proj_4[0:2, 0:2]
    c8_new[4:6, 4:6] = P_proj_4[2:4, 2:4]
    c8_new[0:2, 4:6] = P_proj_4[0:2, 2:4]
    c8_new[4:6, 0:2] = P_proj_4[2:4, 0:2]
    c8_new[np.ix_(idx_ah, idx_ah)] = cov_ah_4

    return m8_new, _nearest_psd(c8_new)


def _bridge_7d_to_8d_2d(x7: np.ndarray, P7: np.ndarray,
                          m8_prev: np.ndarray, c8_prev: np.ndarray,
                          dt: float, w_pos: float, w_vel: float):
    """Convert 7-D pixel-space state to DeepSORT 8-D state (2D fallback).

    In 2D fallback mode: pX = image x, pZ = image y (no projection needed).
    Image-plane velocity: vx = v*cos(phi), vy = v*sin(phi).
    """
    pX, pZ, v, phi, omega, _pY, _vY = x7
    vx, vz = _xz_from_vphi(v, phi)

    a  = m8_prev[2] + m8_prev[6] * dt
    h  = m8_prev[3] + m8_prev[7] * dt
    va, vh = m8_prev[6], m8_prev[7]

    m8_new = np.array([pX, pZ, a, h, vx, vz, va, vh])

    # Jacobian (4x7) for [pX, pZ, vx, vz] w.r.t. x7
    J = np.zeros((4, 7))
    J[0, 0] = 1.0   # d(pX)/d(pX)
    J[1, 1] = 1.0   # d(pZ)/d(pZ)
    J[2, 2] =  np.cos(phi); J[2, 3] = -v * np.sin(phi)
    J[3, 2] =  np.sin(phi); J[3, 3] =  v * np.cos(phi)

    P_xyv = _nearest_psd(J @ P7 @ J.T)

    idx_ah = [2, 3, 6, 7]
    F_ah = np.eye(4); F_ah[0, 2] = dt; F_ah[1, 3] = dt
    std_ah = [w_pos * h, w_pos * h, 1e-5, w_vel * h]
    Q_ah = np.diag(np.square(std_ah))
    cov_ah_4 = F_ah @ c8_prev[np.ix_(idx_ah, idx_ah)] @ F_ah.T + Q_ah

    c8_new = np.eye(8) * 1e-8
    c8_new[0:2, 0:2] = P_xyv[0:2, 0:2]
    c8_new[4:6, 4:6] = P_xyv[2:4, 2:4]
    c8_new[0:2, 4:6] = P_xyv[0:2, 2:4]
    c8_new[4:6, 0:2] = P_xyv[2:4, 0:2]
    c8_new[np.ix_(idx_ah, idx_ah)] = cov_ah_4

    return m8_new, _nearest_psd(c8_new)


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
        max_dist: ignore neighbours beyond this distance (metres or pixels).
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
    """Decompose 32-D mean / 32x32 covariance into IMM components."""
    x_fused  = mean[_SL_FUSED].copy()
    P_fused  = cov[_SL_FUSED, _SL_FUSED].copy()
    x_models = [mean[sl].copy() for sl in _SL_M]
    P_models = [cov[sl, sl].copy()  for sl in _SL_M]
    mu       = mean[_SL_PROB].copy()
    mu       = np.maximum(mu, 1e-8)
    mu      /= mu.sum()
    return x_fused, P_fused, x_models, P_models, mu


def _pack(x_fused, P_fused, x_models, P_models, mu):
    """Assemble 32-D mean and 32x32 covariance from IMM components."""
    mean = np.zeros(_TOTAL)
    cov  = np.eye(_TOTAL) * 1e-8   # tiny diagonal baseline

    mean[_SL_FUSED] = x_fused
    cov[_SL_FUSED, _SL_FUSED] = P_fused

    for i, sl in enumerate(_SL_M):
        mean[sl]    = x_models[i]
        cov[sl, sl] = P_models[i]

    mean[_SL_PROB] = mu
    return mean, cov


# ══════════════════════════════════════════════════════════════════════════════
#  BehavioralIMMFilter  –  main public class
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralIMMFilter:
    """IMM filter — drop-in replacement for BehavioralEKFFilter.

    Works in 3D camera-frame coordinates when depth images are available
    (world_pos provided to initiate/update), or falls back to 2D image-frame
    tracking when depth is not available (world_pos=None).

    Public interface (unchanged from original):
        initiate(measurement, world_pos=None)             -> (mean_32, cov_32x32)
        predict(mean, covariance, other_track_means=None) -> (mean_32, cov_32x32)
        update(mean, covariance, measurement, world_pos=None) -> (mean_32, cov_32x32)
        gating_distance(mean, cov, measurements, ...)     -> ndarray of distances
        project(mean, covariance)                         -> (mean_4, cov_4x4)
    """

    def __init__(
        self,
        dt: float = 1.0,
        mu_init: tuple = (0.6, 0.3, 0.1),
        ca_decel: float = 0.85,
        ca_decel_y: float = 0.9,
        # Camera intrinsics (for 3D → image projection)
        fx: float = 400.0,
        fy: float = 400.0,
        cx: float = 400.0,
        cy: float = 300.0,
        # Set True when depth images are always available (uses world-scale Q)
        # Set False for 2D pixel-frame fallback (uses pixel-scale Q)
        has_depth: bool = False,
    ):
        self._dt        = dt
        self._ca_decel  = ca_decel
        self._ca_decel_y = ca_decel_y
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._has_depth = has_depth

        self._mu_init = np.array(mu_init, dtype=np.float64)
        self._mu_init /= self._mu_init.sum()

        # ── Measurement noise ─────────────────────────────────────────────────
        if has_depth:
            # 3-D measurement noise in metres
            # sigma_XZ = 0.05 m (lateral + depth from depth sensor)
            # sigma_Y  = 0.10 m (vertical; depth noisier for small objects)
            self._R3 = np.diag([0.05**2, 0.05**2, 0.10**2])
            # 2-D fallback noise (used when a frame temporarily has no depth)
            self._R2 = np.diag([0.05**2, 0.05**2])
        else:
            # 2-D pixel-frame measurement noise
            # 0.05 of normalised coords → realistic YOLOv9 box jitter
            self._R3 = np.diag([0.05**2, 0.05**2, 0.10**2])  # kept for API completeness
            self._R2 = np.diag([0.05**2, 0.05**2])

        # ── Markov transition matrix Pi (row i -> col j) ──────────────────────
        self._Pi_base = np.array([
            [0.90, 0.07, 0.03],   # CV → (CV, CT, CA)
            [0.10, 0.85, 0.05],   # CT → (CV, CT, CA)
            [0.15, 0.05, 0.80],   # CA → (CV, CT, CA)
        ], dtype=np.float64)

        # ── Social force parameters ───────────────────────────────────────────
        if has_depth:
            # World frame: distances in metres; typical pedestrian bubble ~1m
            self._sf_max_dist  = 3.0    # metres
            self._sf_threshold = 0.5    # m/s^2 (world frame)
        else:
            # Pixel frame: typical pedestrian bbox ~100px; bubble ~150px
            self._sf_max_dist  = 150.0  # pixels
            self._sf_threshold = 50.0   # pixel force units

        # ── Per-model process noise Q (7x7) ───────────────────────────────────
        # Order: [pX, pZ, v, phi, omega, pY, vY]
        if has_depth:
            # World-frame (metres): pedestrian ~1.5 m/s, dt=1/30 s => ~0.05 m/frame
            self._Q = [
                np.diag([1e-2, 1e-2, 0.03, 0.005, 1e-5, 5e-3, 5e-4])**2,   # CV
                np.diag([1e-2, 1e-2, 0.02, 0.010, 0.05, 5e-3, 5e-4])**2,   # CT
                np.diag([1e-2, 1e-2, 0.10, 0.005, 1e-5, 5e-3, 5e-3])**2,   # CA
            ]
        else:
            # Pixel-frame: same scale as original 5-D IMM (extended with pY/vY rows)
            self._Q = [
                np.diag([1.0, 1.0, 1.5, 2.0, 0.1, 0.5, 0.1])**2,   # CV
                np.diag([1.0, 1.0, 1.9, 3.0, 3.0, 0.5, 0.1])**2,   # CT
                np.diag([1.5, 1.5, 5.0, 2.0, 0.1, 0.5, 0.5])**2,   # CA
            ]

        # DeepSORT relative noise weights (same as original KalmanFilter)
        self._w_pos = 1.0 / 20
        self._w_vel = 1.0 / 160

    # ── Public API ─────────────────────────────────────────────────────────────

    def initiate(self, measurement, world_pos=None):
        """Create a new track from a 4-D detection [x, y, a, h].

        Args:
            measurement: 4-D array [x_pix, y_pix, aspect_ratio, height].
            world_pos:   Optional 3-D array [pX, pZ, pY] in camera frame (metres).
                         If None, 2D pixel-frame tracking is used.

        Returns:
            (mean_32, cov_32x32) packed state.
        """
        z  = np.asarray(measurement, dtype=np.float64)
        h  = z[3]

        # ── Fused 8-D initial state ───────────────────────────────────────────
        x_fused = np.r_[z, np.zeros(4)]
        std_8d  = [
            2 * self._w_pos * h,  2 * self._w_pos * h,  1e-2,  2 * self._w_pos * h,
            10 * self._w_vel * h, 10 * self._w_vel * h, 1e-5, 10 * self._w_vel * h,
        ]
        P_fused = np.diag(np.square(std_8d))

        # ── Per-model 7-D initial states ──────────────────────────────────────
        if world_pos is not None:
            pX, pZ, pY = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])
            # Position known from depth; velocity unknown at first detection
            x7_init = np.array([pX, pZ, 0.0, 0.0, 0.0, pY, 0.0])
            # Position variance from depth accuracy (~0.05m lateral, ~0.1m depth)
            p_pos_xz = (0.05)**2
            p_pos_y  = (0.10)**2
        else:
            # 2D fallback: treat image center as (pX, pZ), pY=0
            x7_init = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0, 0.0])
            p_pos_xz = (2 * self._w_pos * h)**2
            p_pos_y  = 1e-2

        P7_init = np.diag([
            p_pos_xz,                         # pX
            p_pos_xz,                         # pZ
            (10 * self._w_vel * h) ** 2,      # v  (unknown initially)
            0.1,                              # phi (unknown)
            0.01,                             # omega
            p_pos_y,                          # pY
            1e-4,                             # vY
        ])

        x_models = [x7_init.copy() for _ in range(_N)]
        P_models = [P7_init.copy() for _ in range(_N)]

        return _pack(x_fused, P_fused, x_models, P_models, self._mu_init.copy())

    def predict(self, mean, covariance, other_track_means=None):
        """IMM prediction: Mixing -> Mode-conditioned prediction -> Fusion.

        Args:
            mean:              32-D packed mean from previous step.
            covariance:        32x32 packed covariance.
            other_track_means: list of other tracks' 32-D means (for social force).

        Returns:
            (mean_pred, cov_pred): 32-D, 32x32.
        """
        x_fused, P_fused, x_models, P_models, mu = _unpack(mean, covariance)
        dt = self._dt

        # ── Social force ──────────────────────────────────────────────────────
        # Use world positions (pX, pZ) from the CV model sub-state of each track.
        # In 2D fallback mode these are pixel coordinates — social force still works,
        # just in pixel units.
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

        # ── Step 5: Fuse predicted states ─────────────────────────────────────
        x7_fused = sum(c_bar[j] * x_pred[j] for j in range(_N))
        P7_fused = np.zeros((7, 7))
        for j in range(_N):
            dx = x_pred[j] - x7_fused
            P7_fused += c_bar[j] * (P_pred[j] + np.outer(dx, dx))
        P7_fused = _nearest_psd(P7_fused)

        # Convert fused 7-D world state -> 8-D DeepSORT state
        if self._has_depth:
            x_fused_new, P_fused_new = _bridge_7d_to_8d_3d(
                x7_fused, P7_fused, x_fused, P_fused, dt,
                self._w_pos, self._w_vel,
                self._fx, self._fy, self._cx, self._cy
            )
        else:
            x_fused_new, P_fused_new = _bridge_7d_to_8d_2d(
                x7_fused, P7_fused, x_fused, P_fused, dt,
                self._w_pos, self._w_vel
            )

        return _pack(x_fused_new, _nearest_psd(P_fused_new), x_pred, P_pred, c_bar)

    def update(self, mean, covariance, measurement, world_pos=None):
        """IMM update: Mode-conditioned update -> Mode prob update -> Fusion.

        Args:
            mean:        32-D packed mean (output of predict).
            covariance:  32x32 packed covariance.
            measurement: 4-D observation [x_pix, y_pix, aspect_ratio, height].
            world_pos:   Optional 3-D [pX, pZ, pY] in camera frame (metres).
                         If None, 2D fallback update is used.

        Returns:
            (mean_upd, cov_upd): 32-D, 32x32.
        """
        x_fused, P_fused, x_pred, P_pred, mu = _unpack(mean, covariance)
        z = np.asarray(measurement, dtype=np.float64)

        # ── Mode-conditioned update ───────────────────────────────────────────
        x_upd, P_upd, log_liks = [], [], []
        for j in range(_N):
            if world_pos is not None:
                z_world = np.asarray(world_pos, dtype=np.float64)  # [pX, pZ, pY]
                xu, Pu, innov, S = _ekf_update_7d_3d(
                    x_pred[j], P_pred[j], z_world, self._R3)
            else:
                z_xy = z[:2]   # [x_pix, y_pix]
                xu, Pu, innov, S = _ekf_update_7d_2d(
                    x_pred[j], P_pred[j], z_xy, self._R2)
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

        # ── Fused updated estimate ─────────────────────────────────────────────
        x7_fused_upd = sum(mu_new[j] * x_upd[j] for j in range(_N))
        P7_fused_upd = np.zeros((7, 7))
        for j in range(_N):
            dx = x_upd[j] - x7_fused_upd
            P7_fused_upd += mu_new[j] * (P_upd[j] + np.outer(dx, dx))
        P7_fused_upd = _nearest_psd(P7_fused_upd)

        # ── Rebuild 8-D fused state ───────────────────────────────────────────
        a_new, h_new = z[2], z[3]

        if world_pos is not None:
            # 3D mode: project updated world state to image plane
            pX = x7_fused_upd[0]
            pZ = max(float(x7_fused_upd[1]), 0.1)
            pY = x7_fused_upd[5]
            v, phi = x7_fused_upd[2], x7_fused_upd[3]
            vY = x7_fused_upd[6]

            vX_w = v * np.cos(phi)
            vZ_w = v * np.sin(phi)

            u_new     = self._fx * pX / pZ + self._cx
            v_img_new = self._fy * pY / pZ + self._cy
            vx_img    = self._fx * (vX_w * pZ - pX * vZ_w)  / pZ**2
            vy_img    = self._fy * (vY   * pZ - pY * vZ_w)  / pZ**2

            x_fused_new = np.array([
                u_new, v_img_new, a_new, h_new,
                vx_img, vy_img,
                x_fused[6], x_fused[7],   # carry-forward va, vh
            ])

            # 8-D covariance: project 7-D via Jacobian of projection
            J = np.zeros((4, 7))
            J[0, 0] =  self._fx / pZ
            J[0, 1] = -self._fx * pX / pZ**2
            J[1, 5] =  self._fy / pZ
            J[1, 1] = -self._fy * pY / pZ**2
            J[2, 0] = -self._fx * vZ_w / pZ**2
            J[2, 1] =  self._fx * (2*pX*vZ_w - vX_w*pZ) / pZ**3
            J[2, 2] =  self._fx * (np.cos(phi)*pZ - pX*np.sin(phi)) / pZ**2
            J[2, 3] =  self._fx * v * (-np.sin(phi)*pZ - pX*np.cos(phi)) / pZ**2
            J[3, 1] =  self._fy * (2*pY*vZ_w - vY*pZ) / pZ**3
            J[3, 2] = -self._fy * pY * np.sin(phi) / pZ**2
            J[3, 3] = -self._fy * pY * v * np.cos(phi) / pZ**2
            J[3, 5] = -self._fy * vZ_w / pZ**2
            J[3, 6] =  self._fy / pZ

            P_proj_4 = _nearest_psd(J @ P7_fused_upd @ J.T)

            P_fused_new = np.eye(8) * 1e-8
            P_fused_new[0:2, 0:2] = P_proj_4[0:2, 0:2]
            P_fused_new[4:6, 4:6] = P_proj_4[2:4, 2:4]
            P_fused_new[0:2, 4:6] = P_proj_4[0:2, 2:4]
            P_fused_new[4:6, 0:2] = P_proj_4[2:4, 0:2]
            P_fused_new[2:4, 2:4] = np.diag([1e-2**2, (self._w_pos * h_new)**2])
            P_fused_new[6:8, 6:8] = np.diag([1e-5**2, (self._w_vel * h_new)**2])

        else:
            # 2D fallback: pX = image x, pZ = image y (no projection)
            v, phi = x7_fused_upd[2], x7_fused_upd[3]
            vx, vz = _xz_from_vphi(v, phi)
            x_fused_new = np.array([
                x7_fused_upd[0], x7_fused_upd[1],
                a_new, h_new, vx, vz,
                x_fused[6], x_fused[7],
            ])

            J_vy = np.array([
                [np.cos(phi), -v * np.sin(phi)],
                [np.sin(phi),  v * np.cos(phi)],
            ])
            P_fused_new = np.eye(8) * 1e-8
            P_fused_new[0:2, 0:2] = P7_fused_upd[0:2, 0:2]
            P_fused_new[4:6, 4:6] = J_vy @ P7_fused_upd[2:4, 2:4] @ J_vy.T
            P_fused_new[0:2, 4:6] = P7_fused_upd[0:2, 2:4] @ J_vy.T
            P_fused_new[4:6, 0:2] = P_fused_new[0:2, 4:6].T
            P_fused_new[2:4, 2:4] = np.diag([1e-2**2, (self._w_pos * h_new)**2])
            P_fused_new[6:8, 6:8] = np.diag([1e-5**2, (self._w_vel * h_new)**2])

        return _pack(x_fused_new, _nearest_psd(P_fused_new), x_upd, P_upd, mu_new)

    def project(self, mean, covariance):
        """Project to 4-D measurement space [x, y, a, h].

        The fused 8-D state always carries image-plane coordinates, so this
        method is mode-agnostic (no camera intrinsics needed here).
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

        Operates in 4-D image-space [x, y, a, h] — mode-agnostic, since the
        fused 8-D state always contains image-plane coordinates.

        Returns array of length len(measurements).
        """
        mean_proj, cov_proj = self.project(mean, covariance)
        meas = np.asarray(measurements)

        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj  = cov_proj[:2, :2]
            meas      = meas[:, :2]

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
            return np.full(meas.shape[0], 1e5)
