"""Compatibility imports for rendezvous.actuation."""

from rendezvous.actuation import (
    ActuationResult as ActuationResult,
    ReactionWheelResult as ReactionWheelResult,
    ThrusterArray as ThrusterArray,
    translation_thrusters as translation_thrusters,
    educational_attitude_thrusters as educational_attitude_thrusters,
    chaser_rcs as chaser_rcs,
    ReactionWheels as ReactionWheels,
)

def __getattr__(name):
    if name in {"attach_basilisk_rcs", "WheelPointingController"}:
        from rendezvous.basilisk import effectors

        return getattr(effectors, name)
    raise AttributeError(name)
