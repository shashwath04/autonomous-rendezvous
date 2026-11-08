from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ActuationResult:
    thrust: np.ndarray            
    duty_cycle: np.ndarray         
    force_body: np.ndarray        
    torque_body: np.ndarray       
    residual: np.ndarray

class ThrusterArray:
    def __init__(self, positions, directions, max_thrust):
       
        self.positions = np.array(positions, dtype=float, copy=True)
        directions = np.array(directions, dtype=float, copy=True)

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError('positions must have shape (N, 3)')
        
        if directions.shape != self.positions.shape or len(directions) == 0:
            raise ValueError('directions must match nonempty positions')
        
        norms = np.linalg.norm(directions, axis=1)

        if not np.all(np.isfinite(self.positions)) or not np.all(np.isfinite(directions)) or np.any(norms == 0):
            raise ValueError('Geometry must be finite; directions must be nonzero')
        
        self.directions = directions / norms[:, None]

        self.max_thrust = np.broadcast_to(
            np.asarray(max_thrust, dtype=float), (len(directions),)
        ).copy()

        if not np.all(np.isfinite(self.max_thrust)) or np.any(self.max_thrust <= 0):
            raise ValueError('max_thrust must be finite and positive')
        
        self.force_map = self.directions.T
        self.torque_map = np.cross(self.positions, self.directions).T
        self.wrench_map = np.vstack((self.force_map, self.torque_map))

    def apply(self, thrust):
        thrust = np.asarray(thrust, dtype=float).reshape(len(self.max_thrust))

        if not np.all(np.isfinite(thrust)):
            raise ValueError('Thrust commands must be finite')
        
        actual = np.clip(thrust, 0, self.max_thrust)
        
        return actual, self.force_map @ actual, self.torque_map @ actual

    def allocate(self, force_body=None, torque_body=None,force_scale=1.0, torque_scale=1.0, effort_weight=1e-8):

        force = np.zeros(3) if force_body is None else np.asarray(force_body, dtype=float).reshape(3)
        torque = np.zeros(3) if torque_body is None else np.asarray(torque_body, dtype=float).reshape(3)

        requested = np.concatenate((force, torque))
        settings = np.array([force_scale, torque_scale, effort_weight])

        if not np.all(np.isfinite(settings)) or force_scale <= 0 or torque_scale <= 0 or effort_weight <= 0:
            raise ValueError('Scales and effort_weight must be finite and positive')
        
        if not np.all(np.isfinite(requested)):
            raise ValueError('Requested wrench must be finite')

        
        weights = 1 / np.array([force_scale]*3 + [torque_scale]*3)
        matrix = self.wrench_map * self.max_thrust[None, :]
        matrix = np.vstack((weights[:, None]*matrix,
                            np.sqrt(effort_weight)*np.eye(len(self.max_thrust))))

        
        target = np.concatenate((weights*requested, np.zeros(len(self.max_thrust))))
        solution = lsq_linear(matrix, target, bounds=(0, 1), tol=1e-12)


        if not solution.success:
            raise RuntimeError('Thruster allocation failed: ' + solution.message)
        
        thrust, force_actual, torque_actual = self.apply(solution.x*self.max_thrust)

        return ActuationResult(thrust, thrust/self.max_thrust,
                               force_actual, torque_actual,
                               requested - np.concatenate((force_actual, torque_actual)))


def translation_thrusters(max_thrust=1.0):
   
    directions = np.vstack((np.eye(3), -np.eye(3)))
    return ThrusterArray(np.zeros((6, 3)), directions, max_thrust)


def educational_attitude_thrusters(arm=0.4, max_thrust=1.0):
    
    if not np.isfinite(arm) or arm <= 0:
        raise ValueError('arm must be finite and positive')
    positions, directions = [], []
    for position_axis, force_axis in ((1, 2), (2, 0), (0, 1)):
        r, d = np.eye(3)[position_axis]*arm, np.eye(3)[force_axis]
        for sign in (1, -1):
            positions.extend((r, -r))
            directions.extend((sign*d, -sign*d))
    return ThrusterArray(positions, directions, max_thrust)
