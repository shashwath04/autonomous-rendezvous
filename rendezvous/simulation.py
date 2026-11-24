"""Standalone CW/RK4 reference simulation."""

import numpy as np
from scipy.spatial.transform import Rotation

from . import dynamics, utils
from .flight_control import (
    INERTIA,
    MASS,
    RendezvousFlightControl,
    RendezvousHistory,
    initial_attitude,
    simulation_steps,
)


def simulate(n, initial_state, dt, T):
    steps = simulation_steps(dt, T)
    history = RendezvousHistory(steps, dt)
    state = np.asarray(initial_state, dtype=float).reshape(6, 1)
    attitude = initial_attitude().reshape(10, 1)
    flight = RendezvousFlightControl(n, state, dt)

    history.state[0] = state.ravel()
    history.estimate[0] = flight.ekf.x.ravel()
    history.attitude[0] = attitude.ravel()

    for step in range(steps):
        command = flight.command(attitude)
        torque_args = (INERTIA, command.external_torque, command.wheel_torque, n)
        attitude_mid = utils.rk4(dynamics.attitude_derivative, attitude, dt / 2, *torque_args)
        c_mid = Rotation.from_quat(attitude_mid[:4, 0]).as_matrix()
        acceleration = c_mid @ command.force_body / MASS

        state = utils.rk4(
            dynamics.cw_derivative, state, dt, flight.A, flight.B, acceleration[:, None]
        )
        attitude = utils.rk4(dynamics.attitude_derivative, attitude, dt, *torque_args)
        attitude[:4] /= np.linalg.norm(attitude[:4])
        limit = flight.wheels.max_momentum
        attitude[7:10, 0] = np.clip(attitude[7:10, 0], -limit, limit)

        next_step = step + 1
        flight.observe(history.time[next_step], state, attitude[:4, 0], acceleration)
        history.state[next_step] = state.ravel()
        history.estimate[next_step] = flight.ekf.x.ravel()
        history.attitude[next_step] = attitude.ravel()
        history.acceleration[step] = acceleration
        history.wheel_torque[step] = command.wheel_torque
        history.external_torque[step] = command.external_torque

    return history.result()
