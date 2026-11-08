import numpy as np
from scipy.spatial.transform import Rotation
from dynamics import cw_derivative


def forward_euler(A, B, state, control, dt):
    state_dot = cw_derivative(A, B, state, control)
    next_state = state + state_dot * dt
    return next_state

def rk4(derivative, state, dt, *args):
    k1 = derivative(state, *args)
    k2 = derivative(state + 0.5 * dt * k1, *args)
    k3 = derivative(state + 0.5 * dt * k2, *args)
    k4 = derivative(state + dt * k3, *args)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def saturate(value, threshold):
    return np.clip(value, -threshold, threshold)

def quaternion_product(q, p):
    qv, qw = q[:3], q[3]
    pv, pw = p[:3], p[3]

    return np.concatenate((
        qw * pv + pw * qv + np.cross(qv, pv),
        [qw * pw - np.dot(qv, pv)],
    ))

def quaternion_derivative(q, omega, n):
    q = np.asarray(q, dtype=float).reshape(4)
    omega = np.asarray(omega, dtype=float).reshape(3)

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion must be nonzero")

    q_unit = q / norm
    C_orbital_body = Rotation.from_quat(q_unit).as_matrix()

    orbital_rate = np.array([0.0, 0.0, n])

    relative_omega = (
        omega - C_orbital_body.T @ orbital_rate
    )

    omega_quaternion = np.concatenate((relative_omega, [0.0]))

    return 0.5 * quaternion_product(q_unit, omega_quaternion)