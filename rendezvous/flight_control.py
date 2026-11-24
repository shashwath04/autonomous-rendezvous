"""Control and navigation shared by both rendezvous simulations."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import actuation, controllers, dynamics
from .navigation import ExtendedKalmanFilter
from .sensors import FFRF, VBS, PhoenixS

MASS = 150.0
INERTIA = np.diag([20.0, 25.0, 15.0])


def initial_attitude():
    quaternion = Rotation.from_euler("xyz", [20, -15, 30], degrees=True).as_quat()
    return np.concatenate((quaternion, np.deg2rad([2, -1, 1.5]), np.zeros(3)))


@dataclass(frozen=True)
class Control:
    force_body: np.ndarray
    external_torque: np.ndarray
    wheel_torque: np.ndarray


class RendezvousFlightControl:
    """Run sampled sensor fusion and allocate translation and attitude commands."""

    sensor_periods = {"gps": 1.0, "ffrf": 1.0, "vbs": 0.2}

    def __init__(self, n, initial_state, dt):
        self.n = n
        self.dt = dt
        self.translation = actuation.translation_thrusters()
        self.attitude = actuation.educational_attitude_thrusters()
        self.wheels = actuation.ReactionWheels(max_torque=0.02, max_momentum=0.2)
        self.detumbling = True
        self.kp = 0.1 * np.eye(3)
        self.kd = 2.0 * np.eye(3)
        self.c_camera_body = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

        self.A, self.B = dynamics.cw_matrices(n)
        Ad, Bd, self.gain = controllers.discrete_lqr(self.A, self.B, dt)
        identity = np.eye(3)
        process_noise = 1e-8 * np.block([
            [dt**3 / 3 * identity, dt**2 / 2 * identity],
            [dt**2 / 2 * identity, dt * identity],
        ])

        self.gps = PhoenixS(position_std=1.0, velocity_std=0.01, seed=1)
        self.ffrf = FFRF(position_std=0.1, seed=2)
        self.vbs = VBS(angle_std=np.deg2rad(0.05), seed=3)
        self.next_measurement = self.sensor_periods.copy()
        self.ekf = ExtendedKalmanFilter(
            Ad, Bd, process_noise, self.gps.measure(initial_state), self.gps.R
        )

    def command(self, attitude_state):
        q, omega, momentum = np.split(np.asarray(attitude_state).reshape(10), [4, 7])
        c_orbital_body = Rotation.from_quat(q).as_matrix()
        acceleration = (-self.gain @ self.ekf.x).ravel()
        translation = self.translation.allocate(c_orbital_body.T @ (MASS * acceleration))

        rate_error = omega - c_orbital_body.T @ [0, 0, self.n]
        if np.linalg.norm(rate_error) <= np.deg2rad(0.1):
            self.detumbling = False

        if self.detumbling:
            attitude = self.attitude.allocate(torque_body=-self.kd @ rate_error, torque_scale=0.8)
            wheel_torque = np.zeros(3)
        else:
            desired_torque = controllers.attitude_pd(
                q, omega, [0, 0, 0, 1], INERTIA, self.n, self.kp, self.kd
            ) + np.cross(omega, momentum)
            remaining = desired_torque - translation.torque_body
            unload_limit = 0.25 * self.wheels.max_torque
            unload_request = np.clip(-0.05 * momentum, -unload_limit, unload_limit)
            wheels = self.wheels.allocate(remaining - unload_request, momentum, self.dt)
            attitude = self.attitude.allocate(
                torque_body=remaining - wheels.torque_body, torque_scale=0.8
            )
            wheel_torque = self.wheels.allocate(
                remaining - attitude.torque_body, momentum, self.dt
            ).torque_body

        return Control(
            translation.force_body + attitude.force_body,
            translation.torque_body + attitude.torque_body,
            wheel_torque,
        )

    def observe(self, time_s, state, quaternion, acceleration):
        self.ekf.predict(acceleration)
        if self._measurement_due("gps", time_s):
            self.ekf.update_gps(self.gps.measure(state), self.gps.R)
        if self._measurement_due("ffrf", time_s):
            self.ekf.update_ffrf(self.ffrf.measure(state), self.ffrf.R)
        if self._measurement_due("vbs", time_s):
            c_orbital_body = Rotation.from_quat(quaternion).as_matrix()
            c_camera_orbital = self.c_camera_body @ c_orbital_body.T
            target_camera = -c_camera_orbital @ np.asarray(state).reshape(6)[:3]
            self.ekf.update_vbs(self.vbs.measure(target_camera), self.vbs.R, c_camera_orbital)

    def _measurement_due(self, sensor, time_s):
        if time_s + 1e-12 < self.next_measurement[sensor]:
            return False
        self.next_measurement[sensor] += self.sensor_periods[sensor]
        return True


def simulation_steps(dt, duration):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("Duration must be finite and nonnegative")
    if dt > min(RendezvousFlightControl.sensor_periods.values()):
        raise ValueError("dt must be no larger than the shortest sensor period")
    count = duration / dt
    nearest = round(count)
    return nearest if abs(count - nearest) <= 2 * np.spacing(count) else int(count)


class RendezvousHistory:
    """Samples at state times and commands over the intervening intervals."""

    def __init__(self, steps, dt):
        self.time = np.arange(steps + 1) * dt
        self.state = np.empty((steps + 1, 6))
        self.estimate = np.empty_like(self.state)
        self.attitude = np.empty((steps + 1, 10))
        self.acceleration = np.empty((steps, 3))
        self.wheel_torque = np.empty_like(self.acceleration)
        self.external_torque = np.empty_like(self.acceleration)

    def result(self):
        return (
            self.time, self.state, self.acceleration, self.estimate,
            self.attitude, self.wheel_torque, self.external_torque,
        )
