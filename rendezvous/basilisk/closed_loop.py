"""Closed-loop rendezvous with Basilisk spacecraft and wheel dynamics."""

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import extForceTorque, reactionWheelStateEffector, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, simIncludeRW
from scipy.spatial.transform import Rotation

from ..flight_control import (
    INERTIA,
    MASS,
    RendezvousFlightControl,
    RendezvousHistory,
    initial_attitude,
    simulation_steps,
)
from .effectors import write_motor_torque


def _chief_frame(position, velocity):
    radial = position / np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    normal = angular_momentum / np.linalg.norm(angular_momentum)
    basis = np.column_stack((radial, np.cross(normal, radial), normal))
    return basis, angular_momentum / np.dot(position, position)


class _FlightTask(sysModel.SysModel):

    def __init__(self, chief, chaser, wrench, flight, history, wheel_inertia):
        super().__init__()
        self.ModelTag = "RendezvousFlightControl"
        self.chief_state = chief.scStateOutMsg
        self.chaser_state = chaser.scStateOutMsg
        self.wrench = wrench
        self.flight = flight
        self.history = history
        self.wheel_inertia = wheel_inertia
        self.motor_cmd = messaging.ArrayMotorTorqueMsg()
        self.step = 0

    def UpdateState(self, current_time_ns):
        chief = self.chief_state.read()
        chaser = self.chaser_state.read()
        chief_position = np.asarray(chief.r_BN_N)
        basis, frame_rate = _chief_frame(chief_position, np.asarray(chief.v_BN_N))
        displacement = np.asarray(chaser.r_BN_N) - chief_position
        velocity = np.asarray(chaser.v_BN_N) - np.asarray(chief.v_BN_N)
        state = np.concatenate((
            basis.T @ displacement,
            basis.T @ (velocity - np.cross(frame_rate, displacement)),
        ))

        attitude = Rotation.from_matrix(basis.T @ Rotation.from_mrp(chaser.sigma_BN).as_matrix())
        quaternion = attitude.as_quat()
        momentum = self.wheel_inertia * np.asarray(self.wheel_speed.getState()).ravel()
        history = self.history
        step = self.step
        if step:
            previous = Rotation.from_quat(history.attitude[step - 1, :4])
            if np.dot(history.attitude[step - 1, :4], quaternion) < 0:
                quaternion = -quaternion
            midpoint = previous * Rotation.from_rotvec((previous.inv() * attitude).as_rotvec() / 2)
            acceleration = midpoint.apply(self.previous_force) / MASS
            history.acceleration[step - 1] = acceleration
            self.flight.observe(history.time[step], state, quaternion, acceleration)

        history.state[step] = state
        history.estimate[step] = self.flight.ekf.x.ravel()
        history.attitude[step] = np.concatenate((quaternion, chaser.omega_BN_B, momentum))

        if step < len(history.wheel_torque):
            
            command_attitude = history.attitude[step].copy()
            limit = self.flight.wheels.max_momentum
            command_attitude[7:] = np.clip(momentum, -limit, limit)
            command = self.flight.command(command_attitude)
            self.wrench.extForce_B = command.force_body.tolist()
            self.wrench.extTorquePntB_B = command.external_torque.tolist()
            write_motor_torque(
                self.motor_cmd, -command.wheel_torque, current_time_ns, self.moduleID
            )
            history.wheel_torque[step] = command.wheel_torque
            history.external_torque[step] = command.external_torque
            self.previous_force = command.force_body

        self.step += 1


def run_rendezvous(n, initial_state, dt, T):
    
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Mean motion must be finite and positive")
    state = np.asarray(initial_state, dtype=float).reshape(6)
    if not np.all(np.isfinite(state)):
        raise ValueError("Initial state must be finite")
    simulation_steps(dt, T)
    period_ns = macros.sec2nano(dt)
    if period_ns < 1:
        raise ValueError("Basilisk requires a step of at least one nanosecond")
    dt = period_ns * macros.NANO2SEC
    steps = simulation_steps(dt, T)
    history = RendezvousHistory(steps, dt)
    flight = RendezvousFlightControl(n, state, dt)

    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("dynamics")
    task_name = "rendezvous"
    process.addTask(sim.CreateNewTask(task_name, period_ns))

    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True
    radius = np.cbrt(earth.mu / n**2)
    position = np.array([radius, 0, 0])
    velocity = np.array([0, n * radius, 0])

    chief = spacecraft.Spacecraft()
    chief.ModelTag = "Chief"
    chief.hub.r_CN_NInit = position.tolist()
    chief.hub.v_CN_NInit = velocity.tolist()

    chaser = spacecraft.Spacecraft()
    chaser.ModelTag = "Chaser"
    attitude = initial_attitude()
    chaser.hub.r_CN_NInit = (position + state[:3]).tolist()
    chaser.hub.v_CN_NInit = (velocity + state[3:] + np.cross([0, 0, n], state[:3])).tolist()
    chaser.hub.sigma_BNInit = Rotation.from_quat(attitude[:4]).as_mrp().tolist()
    chaser.hub.omega_BN_BInit = attitude[4:7].tolist()

    for vehicle in (chief, chaser):
        vehicle.hub.mHub = MASS
        vehicle.hub.IHubPntBc_B = INERTIA.tolist()
        gravity.addBodiesTo(vehicle)
        sim.AddModelToTask(task_name, vehicle, ModelPriority=30)

    wheels = reactionWheelStateEffector.ReactionWheelStateEffector()
    factory = simIncludeRW.rwFactory()
    wheel_inertia = np.empty(3)
    for index, axis in enumerate(np.eye(3)):
        wheel = factory.create(
            "custom", axis.tolist(), RWModel=messaging.BalancedWheels,
            Omega_max=6000.0, maxMomentum=float(flight.wheels.max_momentum[index]),
            u_max=float(flight.wheels.max_torque[index]),
        )
        wheel_inertia[index] = wheel.Js
    factory.addToSpacecraft("ChaserWheels", wheels, chaser)

    wrench = extForceTorque.ExtForceTorque()
    chaser.addDynamicEffector(wrench)
    control = _FlightTask(chief, chaser, wrench, flight, history, wheel_inertia)
    wheels.rwMotorCmdInMsg.subscribeTo(control.motor_cmd)

    sim.AddModelToTask(task_name, control, ModelPriority=20)
    sim.AddModelToTask(task_name, wheels, ModelPriority=10)

    sim.InitializeSimulation()
    control.wheel_speed = chaser.dynManager.getStateObject(wheels.nameOfReactionWheelOmegasState)
    sim.ConfigureStopTime(steps * period_ns)
    sim.ExecuteSimulation()
    if control.step != steps + 1:
        raise RuntimeError("Basilisk did not produce the requested number of samples")
    return history.result()
