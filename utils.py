import numpy as np
from dynamics import cw_derivative

def forward_euler(A, B, state, control, dt):
    state_dot = cw_derivative(A, B, state, control)
    next_state = state + state_dot * dt
    return next_state

def rk4(A, B, state, control, dt):
    k1 = cw_derivative(A, B, state, control)
    k2 = cw_derivative(A, B, state + 0.5 * k1 * dt, control)
    k3 = cw_derivative(A, B, state + 0.5 * k2 * dt, control)
    k4 = cw_derivative(A, B, state + k3 * dt, control)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state

def saturate(value, threshold):
    return np.clip(value, -threshold, threshold)

