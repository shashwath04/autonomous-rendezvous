import numpy as np

from Basilisk.simulation import spacecraft, dragDynamicEffector, exponentialAtmosphere, reactionWheelStateEffector
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, simIncludeRW
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path
from Basilisk.architecture import messaging, sysModel

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

        payload = messaging.ArrayMotorTorqueMsgPayload()
        commands = list(payload.motorTorque)
        commands[:3] = motor_torque.tolist()
        payload.motorTorque = commands
        self.motor_cmd.write(payload, current_time_ns, self.moduleID)


def run_orbit(*, radius_m_chief=6_777_000.0, radius_m_chaser=6_777_010.0, dt_s = 0.1, duration_s = 60.0, wheel_torque_test=False):
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

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()

    return {
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
    }


