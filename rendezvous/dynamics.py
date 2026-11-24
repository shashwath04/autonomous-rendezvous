import numpy as np

from . import utils


def orbital_rate(a, mu):
    return np.sqrt(mu / a**3)


def cw_matrices(n):
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3 * n ** 2, 0, 0, 0, 2 * n, 0],
        [0, 0, 0, -2 * n, 0, 0],
        [0, 0, -n ** 2, 0, 0, 0]
    ])

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return A, B


def cw_derivative(state, A, B, u):
    return A @ state + B @ u


def angular_acceleration(inertia, omega, torque):
    inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
    omega = np.asarray(omega, dtype=float).reshape(3)
    torque = np.asarray(torque, dtype=float).reshape(3)

    return np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))

def attitude_derivative(state, inertia, external_torque, wheel_torque, n):
    state = np.asarray(state, dtype=float).reshape(10)

    q = state[:4]
    omega = state[4:7]
    wheel_momentum = state[7:10]

    external_torque = np.asarray(external_torque, dtype=float).reshape(3)
    wheel_torque = np.asarray(wheel_torque, dtype=float).reshape(3)

    q_dot = utils.quaternion_derivative(q, omega, n)
    total_momentum = inertia @ omega + wheel_momentum
    omega_dot = np.linalg.solve(
        inertia, external_torque + wheel_torque - np.cross(omega, total_momentum)
    )
    momentum_dot = -wheel_torque

    return np.concatenate((q_dot, omega_dot, momentum_dot)).reshape(10, 1)
