import numpy as np

import dynamics
import utils
import controllers
import actuation

from navigation import ExtendedKalmanFilter
from sensors import PhoeinxS, FFRF, VBS

def simulate(n, initial_state, dt, T):
    if dt <= 0 or T < 0:
        raise ValueError("dt must be positive and T must be nonnegative")

    state = np.asarray(initial_state, dtype=float).reshape(-1, 1)
    # u = np.zeros((3, 1))

    mass = 150.0
    thrusters = actuation.translation_thrusters(max_thrust=1.0)

    total_steps = int(T / dt)
    state_size = state.size

    t = np.arange(total_steps + 1) * dt
    states = np.zeros((total_steps + 1, state_size))
    control_states = np.zeros((total_steps + 1, 3))
    states[0] = state.ravel()

    A, B = dynamics.cw_matrices(n)
    Ad, Bd, K = controllers.discrete_lqr(A, B, dt)

    # Defining the sensors

    gps = PhoeinxS(position_std=1.0, velocity_std=1.0, seed=1)
    ffrf = FFRF(position_std=0.1, seed=2)
    vbs = VBS(angle_std=np.deg2rad(0.05), seed=3)

    C_camera_orbital = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ])

    initial_estimate = gps.measure(state)
    P0 = gps.R.copy()

    acceleration_psd = 1e-8 
    I3 = np.eye(3)
    Q_process = acceleration_psd * np.block([
        [(dt**3 / 3) * I3, (dt**2 / 2) * I3],
        [(dt**2 / 2) * I3, dt * I3],
    ])

    ekf = ExtendedKalmanFilter(Ad, Bd, Q_process, initial_estimate, P0)
    estimated_states = np.zeros_like(states)
    estimated_states[0] = ekf.x.ravel()

    # Measurement Periods

    periods = {"gps": 1.0, "ffrf": 1.0, "vbs": 0.2}
    next_measurement = periods.copy()

    # Need to make sure similation step time is smaller than the samellest sensor measurement sample time
    if dt > min(periods.values()):
        raise ValueError("dt must be no larger than the shortest sensor period")

    for step in range(total_steps):

        u = -K @ ekf.x

        resultant_thrust = thrusters.allocate(
            force_body = (mass * u).ravel()
        )

        u = (resultant_thrust.force_body / mass).reshape(3, 1)

        state = utils.rk4(A, B, state, u, dt)

        # Do state estimation:

        ekf.predict(u)
        current_time = t[step + 1]

        if current_time + 1e-12 >= next_measurement["gps"]:
            ekf.update_gps(gps.measure(state), gps.R)
            next_measurement["gps"] += periods["gps"]

        if current_time + 1e-12 >= next_measurement["ffrf"]:
            ekf.update_ffrf(ffrf.measure(state), ffrf.R)
            next_measurement["ffrf"] += periods["ffrf"]

        if current_time + 1e-12 >= next_measurement["vbs"]:
            target_camera = -C_camera_orbital @ state[:3]
            measurement = vbs.measure(target_camera)

            if measurement is not None:
                ekf.update_vbs(
                    measurement,
                    vbs.R,
                    C_camera_orbital,
                )

            next_measurement["vbs"] += periods["vbs"]


        states[step + 1] = state.ravel()
        estimated_states[step + 1] = ekf.x.ravel()
        control_states[step] = u.ravel()

    return t, states, control_states, estimated_states