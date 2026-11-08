import numpy as np
from scipy.linalg import expm, solve_discrete_are


def discrete_lqr(A, B, dt):
    nx, nu = B.shape

    augmented = np.zeros((nx + nu, nx + nu))
    augmented[:nx, :nx] = A
    augmented[:nx, nx:] = B

    discrete = expm(augmented * dt)
    Ad = discrete[:nx, :nx]
    Bd = discrete[:nx, nx:]

    state_scales = np.array([10, 10, 10, 0.1, 0.1, 0.1])
    control_scales = np.array([0.01, 0.01, 0.01])

    Q = np.diag(1.0 / state_scales**2)
    R = np.diag(1.0 / control_scales**2)

    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(
        R + Bd.T @ P @ Bd,
        Bd.T @ P @ Ad,
    )

    return Ad, Bd, K