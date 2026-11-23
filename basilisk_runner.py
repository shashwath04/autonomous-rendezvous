import numpy as np

from Basilisk.simulation import spacecraft, dragDynamicEffector, exponentialAtmosphere, reactionWheelStateEffector, thrusterDynamicEffector
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, simIncludeRW, simIncludeThruster
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path
from Basilisk.architecture import messaging

from actuation import WheelPointingController, ThrusterArray, translation_thrusters, educational_attitude_thrusters


def run_orbit(
    *,
    radius_m_chief=6_777_000.0,
    radius_m_chaser=6_777_010.0,
    dt_s=0.1,
    duration_s=60.0,
    wheel_torque_test=False,
    rcs_wrench_body=None,
    rcs_pulse_window_s=1.0,
):
    """Run the orbit, optionally firing one allocated body-frame RCS pulse."""
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess('dynamics')
    task_name = 'propagate'
    process.addTask(sim.CreateNewTask(task_name, macros.sec2nano(dt_s)))

    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True

    j2_file = get_path(DataFile.LocalGravData.GGM03S_J2_only)
    earth.useSphericalHarmonicsGravityModel(str(j2_file), 2)

    atmosphere = exponentialAtmosphere.ExponentialAtmosphere()
    atmosphere.ModelTag = "EarthAtmosphere"
    atmosphere.planetRadius = 6_378_136.6
    atmosphere.scaleHeight = 60_000.0

    reference_altitude_m = 400_000.0
    reference_density = 4.0e-12
    atmosphere.baseDensity = reference_density * np.exp(reference_altitude_m / atmosphere.scaleHeight)

    chief_area = 1.0
    chaser_area = 2.0

    chief = spacecraft.Spacecraft()
    chief.ModelTag = "Chief"
    chief.hub.mHub = 150.0
    chief.hub.IHubPntBc_B = [[20, 0, 0], [0, 25, 0], [0, 0, 15]]

    chaser = spacecraft.Spacecraft()
    chaser.ModelTag = "Chaser"
    chaser.hub.mHub = 150.0
    chaser.hub.IHubPntBc_B = [[20, 0, 0], [0, 25, 0], [0, 0, 15]]

    chief_circular_speed = np.sqrt(earth.mu / radius_m_chief)
    chief.hub.r_CN_NInit = [[radius_m_chief], [0.0], [0.0]]
    chief.hub.v_CN_NInit = [[0.0],[chief_circular_speed],[0.0]]
    gravity.addBodiesTo(chief)

    chaser_circular_speed = np.sqrt(earth.mu / radius_m_chaser)
    chaser.hub.r_CN_NInit = [[radius_m_chaser], [0.0], [0.0]]
    chaser.hub.v_CN_NInit = [[0.0], [chaser_circular_speed], [0.0]]
    gravity.addBodiesTo(chaser)

    # I need to attache this reaction wheel to the thingamabob now
    max_motor_torque = 0.05

    wheel_factory = simIncludeRW.rwFactory()
    wheel_array = reactionWheelStateEffector.ReactionWheelStateEffector()

    for axis in np.eye(3):
        wheel_factory.create(
            "Honeywell_HR16",
            axis.tolist(),
            RWModel=messaging.BalancedWheels,
            Omega = 0.0,
            Omega_max = 6000.0,
            maxMomentum = 50.0,
            u_max = max_motor_torque,
        )

    wheel_factory.addToSpacecraft("ChaserWheels", wheel_array, chaser)

    if not wheel_torque_test:
        chaser.hub.sigma_BNInit = [[0.02], [-0.03], [0.01]]

    controller = WheelPointingController(
        chaser.scStateOutMsg,
        max_motor_torque,
        torque_test=wheel_torque_test
    )

    wheel_array.rwMotorCmdInMsg.subscribeTo(controller.motor_cmd)

    wheel_recorder = wheel_array.rwSpeedOutMsg.recorder()
    motor_recorder = controller.motor_cmd.recorder()

    # I need to add the thrust allocation stuff

    translation = translation_thrusters(max_thrust = 0.2)
    attitude = educational_attitude_thrusters(arm = 0.4, max_thrust=0.2)

    rcs_geometry = ThrusterArray(
        positions = np.vstack((translation.positions, attitude.positions)),
        directions = np.vstack((translation.directions, attitude.directions)),
        max_thrust=np.concatenate((translation.max_thrust, attitude.max_thrust))
    )

    rcs = thrusterDynamicEffector.ThrusterDynamicEffector()
    thruster_factory = simIncludeThruster.thrusterFactory()

    for position, direction, max_thrust in zip(rcs_geometry.positions, rcs_geometry.directions, rcs_geometry.max_thrust):
        thruster_factory.create(
            "Blank_Thruster",
            position.tolist(),
            direction.tolist(),
            MaxThrust = float(max_thrust),
            steadyIsp = 60.0,
        )

    thruster_factory.addToSpacecraft("ChaserRCS", rcs, chaser)

    atmosphere.addSpacecraftToModel(chief.scStateOutMsg)
    atmosphere.addSpacecraftToModel(chaser.scStateOutMsg)

    spacecrafts = ((chief, chief_area), (chaser, chaser_area))
    drag_effectors = []

    for i, (vehicle, area) in enumerate(spacecrafts):
        drag = dragDynamicEffector.DragDynamicEffector()
        drag.ModelTag = f"{vehicle.ModelTag}Drag"
        drag.coreParams.projectedArea = area
        drag.coreParams.dragCoeff = 2.2
        drag.atmoDensInMsg.subscribeTo(atmosphere.envOutMsgs[i])

        vehicle.addDynamicEffector(drag)
        sim.AddModelToTask(task_name, drag, ModelPriority=20)
        drag_effectors.append(drag)

    chief_recorder = chief.scStateOutMsg.recorder()
    chaser_recorder = chaser.scStateOutMsg.recorder()


    sim.AddModelToTask(task_name, chief, ModelPriority=10)
    sim.AddModelToTask(task_name, chief_recorder, ModelPriority=0)
    sim.AddModelToTask(task_name, chaser, ModelPriority=10)
    sim.AddModelToTask(task_name, chaser_recorder, ModelPriority=0)
    sim.AddModelToTask(task_name, atmosphere, ModelPriority=30)
    sim.AddModelToTask(task_name, controller, ModelPriority=25)
    sim.AddModelToTask(task_name, wheel_array, ModelPriority=20)
    sim.AddModelToTask(task_name, wheel_recorder, ModelPriority=0)
    sim.AddModelToTask(task_name, motor_recorder, ModelPriority=0)
    sim.AddModelToTask(task_name, rcs, ModelPriority=20)

    rcs_cmd = messaging.THRArrayOnTimeCmdMsg()
    rcs.cmdsInMsg.subscribeTo(rcs_cmd)

    rcs_recorders = [msg.recorder() for msg in rcs.thrusterOutMsgs]
    for recorder in rcs_recorders:
        sim.AddModelToTask(task_name, recorder, ModelPriority=0)

    sim.InitializeSimulation()

    payload = messaging.THRArrayOnTimeCmdMsgPayload()
    on_times = list(payload.OnTimeRequest)

    if rcs_wrench_body is not None:
        wrench = np.asarray(rcs_wrench_body, dtype=float).reshape(6)
        allocation = rcs_geometry.allocate(
            force_body=wrench[:3],
            torque_body=wrench[3:],
        )

        num_jets = len(rcs_geometry.max_thrust)
        on_times[:num_jets] = (
            allocation.duty_cycle * rcs_pulse_window_s
        ).tolist()

    payload.OnTimeRequest = on_times
    rcs_cmd.write(payload)
    
    sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()

    result = {
        "time_s_chief": chief_recorder.times() * macros.NANO2SEC,
        "position_m_chief": np.asarray(chief_recorder.r_BN_N),
        "velocity_mps_chief": np.asarray(chief_recorder.v_BN_N),
        "time_s_chaser": chaser_recorder.times() * macros.NANO2SEC,
        "position_m_chaser": np.asarray(chaser_recorder.r_BN_N),
        "velocity_mps_chaser": np.asarray(chaser_recorder.v_BN_N),
        "sigma_BN_chaser": np.asarray(chaser_recorder.sigma_BN),
        "omega_BN_B_chaser": np.asarray(chaser_recorder.omega_BN_B),
        "wheel_speed_rad_s": np.asarray(wheel_recorder.wheelSpeeds)[:, :3],
        "wheel_motor_torque_nm": np.asarray(motor_recorder.motorTorque)[:, :3],
        "rcs_force_body_n": np.sum([rec.thrustForce_B for rec in rcs_recorders], axis=0),
        "rcs_torque_body_nm": np.sum([rec.thrustTorquePntB_B for rec in rcs_recorders], axis=0),
    }
    return result


