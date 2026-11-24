"""Basilisk actuators and command message helpers."""

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.utilities import simIncludeThruster


def make_rcs(vehicle, array):
    """Attach a thruster dynamic effector using the allocator geometry."""
    effector = thrusterDynamicEffector.ThrusterDynamicEffector()
    factory = simIncludeThruster.thrusterFactory()
    for position, direction, limit in zip(
        array.positions, array.directions, array.max_thrust
    ):
        factory.create(
            "Blank_Thruster",
            position.tolist(),
            direction.tolist(),
            MaxThrust=float(limit),
            steadyIsp=60.0,  # s
        )
    factory.addToSpacecraft("ChaserRCS", effector, vehicle)
    return effector


def pulse_payload(array, allocation, pulse_window_s):
    if not np.isfinite(pulse_window_s) or pulse_window_s <= 0:
        raise ValueError("RCS pulse window must be finite and positive")
    payload = messaging.THRArrayOnTimeCmdMsgPayload()
    on_times = list(payload.OnTimeRequest)
    count = len(array.max_thrust)
    if count > len(on_times):
        raise ValueError("Thruster array exceeds Basilisk's command capacity")
    if allocation is not None:
        on_times[:count] = (allocation.duty_cycle * pulse_window_s).tolist()
    payload.OnTimeRequest = on_times
    return payload


def attach_basilisk_rcs(vehicle, array, allocation, pulse_window_s):
    """Attach a thruster cluster and prepare one pulse command."""
    payload = pulse_payload(array, allocation, pulse_window_s)
    effector = make_rcs(vehicle, array)
    command = messaging.THRArrayOnTimeCmdMsg()
    effector.cmdsInMsg.subscribeTo(command)
    return effector, command, payload


def write_motor_torque(message, motor_torque, time_ns, module_id):
    payload = messaging.ArrayMotorTorqueMsgPayload()
    commands = list(payload.motorTorque)
    commands[:len(motor_torque)] = motor_torque.tolist()
    payload.motorTorque = commands
    message.write(payload, time_ns, module_id)


class WheelPointingController(sysModel.SysModel):
    def __init__(self, state_msg, max_motor_torque, *, torque_test=False):
        super().__init__()
        self.ModelTag = "ChaserWheelPointing"
        self.state_msg = state_msg
        self.max_motor_torque = max_motor_torque
        self.torque_test = torque_test
        self.motor_cmd = messaging.ArrayMotorTorqueMsg()

    def UpdateState(self, current_time_ns):
        if self.torque_test:
            motor_torque = np.full(3, 0.01)
        else:
            state = self.state_msg.read()
            sigma = np.asarray(state.sigma_BN).reshape(3)
            omega = np.asarray(state.omega_BN_B).reshape(3)

            body_torque = -0.8 * sigma - 4.0 * omega
            motor_torque = -body_torque

        motor_torque = np.clip(motor_torque, -self.max_motor_torque, self.max_motor_torque)

        write_motor_torque(self.motor_cmd, motor_torque, current_time_ns, self.moduleID)
