import numpy as np

def orbital_rate(a, mu):
    n = np.sqrt(mu / a ** 3)
    return n

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

def cw_derivative(A, B, x, u):
    state_dot = A @ x + B @ u
    return state_dot

