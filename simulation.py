import numpy as np
from scipy.spatial.transform import Rotation

import dynamics
import utils
import controllers
import actuation

from navigation import ExtendedKalmanFilter
from sensors import PhoeinxS, FFRF, VBS

# TODO: ADD (3 * sigma) BOUNDS: SAVE EKF COVARIANCE AT EACH TIMESTEP

def simulate(n, initial_state, dt, T):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(T) or T < 0:
        raise ValueError("T must be finite and nonnegative")

    state = np.asarray(initial_state, dtype=float).reshape(6, 1)

    mass = 150.0
    inertia = np.diag([20.0, 25.0, 15.0])

    translation_thrusters = actuation.translation_thrusters(max_thrust=1.0)
    attitude_thrusters = actuation.educational_attitude_thrusters(arm=0.4,max_thrust=1.0,)

    wheels = actuation.ReactionWheels(max_torque=0.02,max_momentum=0.2,)
    q_initial = Rotation.from_euler("xyz", [20.0, -15.0, 30.0], degrees=True).as_quat()
    omega_initial = np.deg2rad([2.0, -1.0, 1.5])
    attitude_state = np.concatenate((q_initial, omega_initial, np.zeros(3),)).reshape(10, 1)

    q_desired = np.array([0.0, 0.0, 0.0, 1.0])

    Kp = np.diag([0.1, 0.1, 0.1])
    Kd = np.diag([2.0, 2.0, 2.0])

    detumbling = True
    detumble_exit_rate = np.deg2rad(0.1)
    detumble_gain = np.diag([2.0, 2.0, 2.0])
    
    unloading_gain = 0.05  

    A, B = dynamics.cw_matrices(n)
    Ad, Bd, K = controllers.discrete_lqr(A, B, dt)

    gps = PhoeinxS(position_std=1.0,velocity_std=0.01,seed=1)
    ffrf = FFRF(position_std=0.1, seed=2)
    vbs = VBS(angle_std=np.deg2rad(0.05), seed=3)

    C_camera_body = np.array([
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

    ekf = ExtendedKalmanFilter(
        Ad, Bd, Q_process, initial_estimate, P0
    )

    periods = {"gps": 1.0, "ffrf": 1.0, "vbs": 0.2}
    next_measurement = periods.copy()

    if dt > min(periods.values()):
        raise ValueError(
            "dt must be no larger than the shortest sensor period"
        )

  
    total_steps = int(T / dt)
    t = np.arange(total_steps + 1) * dt

    states = np.zeros((total_steps + 1, 6))
    estimated_states = np.zeros_like(states)
    attitude_states = np.zeros((total_steps + 1, 10))

    control_states = np.zeros((total_steps, 3))
    wheel_torques = np.zeros((total_steps, 3))
    external_torques = np.zeros((total_steps, 3))

    states[0] = state.ravel()
    estimated_states[0] = ekf.x.ravel()
    attitude_states[0] = attitude_state.ravel()

    for step in range(total_steps):
        q = attitude_state[:4, 0]
        omega = attitude_state[4:7, 0]
        wheel_momentum = attitude_state[7:10, 0]

        C_orbital_body = Rotation.from_quat(q).as_matrix()

       
        desired_acceleration = (-K @ ekf.x).ravel()
        desired_force_body = (C_orbital_body.T @ (mass * desired_acceleration))

        translation_result = translation_thrusters.allocate(force_body=desired_force_body)

        omega_reference = (C_orbital_body.T @ np.array([0.0, 0.0, n]))
        relative_rate = omega - omega_reference

        if detumbling:
            if np.linalg.norm(relative_rate) <= detumble_exit_rate:
                detumbling = False

        if detumbling:
            attitude_result = attitude_thrusters.allocate(
                torque_body=-detumble_gain @ relative_rate,
                force_body=np.zeros(3),
                torque_scale=0.8,
            )

            wheel_result = wheels.allocate(np.zeros(3), wheel_momentum, dt)

        else:
            desired_total_torque = controllers.attitude_pd(q, omega, q_desired, inertia, n, Kp, Kd)
            desired_total_torque += np.cross(omega, wheel_momentum)

            unload_request = np.clip(-unloading_gain * wheel_momentum, -0.25 * wheels.max_torque, 0.25 * wheels.max_torque)
            wheel_result = wheels.allocate(
                desired_total_torque - translation_result.torque_body - unload_request, wheel_momentum, dt)

            external_request = (desired_total_torque - translation_result.torque_body - wheel_result.torque_body)
            attitude_result = attitude_thrusters.allocate(torque_body=external_request, force_body=np.zeros(3), torque_scale=0.8)

            wheel_result = wheels.allocate(desired_total_torque - translation_result.torque_body - attitude_result.torque_body, wheel_momentum, dt)

        force_body = (translation_result.force_body + attitude_result.force_body)

        external_torque = (translation_result.torque_body + attitude_result.torque_body)
        wheel_torque = wheel_result.torque_body

       
        attitude_mid = utils.rk4(
            dynamics.attitude_derivative,
            attitude_state,
            dt / 2,
            inertia, external_torque, wheel_torque, n,
        )
        attitude_mid[:4] /= np.linalg.norm(attitude_mid[:4])

        C_mid = Rotation.from_quat(
            attitude_mid[:4, 0]
        ).as_matrix()

        u = (C_mid @ force_body / mass).reshape(3, 1)

        state = utils.rk4(
            dynamics.cw_derivative,
            state,
            dt,
            A, B, u,
        )

        attitude_state = utils.rk4(
            dynamics.attitude_derivative,
            attitude_state,
            dt,
            inertia, external_torque, wheel_torque, n,
        )
        attitude_state[:4] /= np.linalg.norm(attitude_state[:4])

       
        attitude_state[7:10, 0] = np.clip(wheel_result.momentum_next, -wheels.max_momentum, wheels.max_momentum)

        ekf.predict(u)
        current_time = t[step + 1]

        if current_time + 1e-12 >= next_measurement["gps"]:
            ekf.update_gps(gps.measure(state), gps.R)
            next_measurement["gps"] += periods["gps"]

        if current_time + 1e-12 >= next_measurement["ffrf"]:
            ekf.update_ffrf(ffrf.measure(state), ffrf.R)
            next_measurement["ffrf"] += periods["ffrf"]

        if current_time + 1e-12 >= next_measurement["vbs"]:
            C_orbital_body = Rotation.from_quat(attitude_state[:4, 0]).as_matrix()
            C_camera_orbital = (C_camera_body @ C_orbital_body.T)

            target_camera = -C_camera_orbital @ state[:3]
            measurement = vbs.measure(target_camera)

            if measurement is not None:
                ekf.update_vbs(measurement, vbs.R, C_camera_orbital)

            next_measurement["vbs"] += periods["vbs"]

        states[step + 1] = state.ravel()
        estimated_states[step + 1] = ekf.x.ravel()
        attitude_states[step + 1] = attitude_state.ravel()

        control_states[step] = u.ravel()
        wheel_torques[step] = wheel_torque
        external_torques[step] = external_torque

    return t, states, control_states, estimated_states, attitude_states, wheel_torques, external_torques