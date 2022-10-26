# coding: utf8

import numpy as np  # Numpy library
import pybullet_data
import example_robot_data # Functions to load the SOLO quadruped
import pinocchio as pin

import pybullet as p  # PyBullet simulator


def configure_simulation(dt, enableGUI):
    global jointTorques
    # Load the robot for Pinocchio
    solo = example_robot_data.load("solo12", False)
    # solo.initDisplay(loadModel=True)

    # Start the client for PyBullet
    if enableGUI:
        physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    else:
        physicsClient = p.connect(p.DIRECT)  # noqa
    # p.GUI for graphical version
    # p.DIRECT for non-graphical version

    # Set gravity (disabled by default)
    p.setGravity(0, 0, -9.81)

    # Load horizontal plane for PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    global planeId
    planeId = p.loadURDF("plane.urdf")
    print("PLANE DYNAMICS")
    p.changeDynamics(planeId, -1, lateralFriction=1.)
    # p.changeDynamics(planeId, -1, lateralFriction=0.9, restitution = -1, contactDamping = 100., contactStiffness= 50.)
    elements_ = [
        "mass", "lateral_friction", "local_inertia_diag", "local inertial pos", "local inertial orn", "restitution",
        "rolling friction", "spinning friction", "contact damping", "contact stiffness", "body type",
        "collision margin"
    ]

    for k,elt in enumerate(p.getDynamicsInfo(planeId, -1)):
        elem = elements_[k] + " : "
        print(elem , elt)

    # Load the robot for PyBullet
    robotStartPos = [0, 0, 0.35]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.setAdditionalSearchPath(example_robot_data.path.EXAMPLE_ROBOT_DATA_MODEL_DIR + "/solo_description/robots")
    robotId = p.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

    JointFootId = [3,7,11,15]
    for id_ in JointFootId:
        pass
        # p.changeDynamics(robotId, id_, lateralFriction=0.9, restitution = -1, contactDamping = 100., contactStiffness= 50.)
    # ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    toprint = "\n\nSOLO JOINT " + str(JointFootId[0]) + " DYNAMICS"
    print(toprint)
    for k,elt in enumerate(p.getDynamicsInfo(robotId, JointFootId[0])):
        elem = elements_[k] + " : "
        print(elem , elt)

    # Set time step of the simulation
    dt = 0.001
    p.setTimeStep(dt)
    # realTimeSimulation = True # If True then we will sleep in the main loop to have a frequency of 1/dt

    # Disable default motor control for revolute joints
    # revoluteJointIndices = [0, 1, 3, 4, 6, 7, 9, 10]
    revoluteJointIndices =  [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    p.setJointMotorControlArray(robotId,
                                jointIndices=revoluteJointIndices,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=[0.0 for m in revoluteJointIndices],
                                forces=[0.0 for m in revoluteJointIndices])

    # Enable torque control for revolute joints
    jointTorques = [0.0 for m in revoluteJointIndices]
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation for initialization
    p.stepSimulation()

    # Pause simulation for camera recording
    # from IPython import embed
    # embed()

    return robotId, solo, revoluteJointIndices


# Function to get the position/velocity of the base and the angular position/velocity of all joints
def getPosVelJoints(robotId, revoluteJointIndices):

    jointStates = p.getJointStates(robotId, revoluteJointIndices)  # State of all joints
    baseState = p.getBasePositionAndOrientation(robotId)  # Position of the free flying base
    baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base

    # Reshaping data into q and qdot
    q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
                   np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
    qdot = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(),
                      np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))

    return q, qdot


# Order of the forces
forceSensors = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

# Function to get contact forces from the simulator
def getContactForces(robotId, revoluteJointIndices):
    """ Compute contact forces.
    """
    contact_list = p.getContactPoints(robotId)
    f = {name: [0, pin.Force.Zero(), 1] for name in forceSensors}
    s = {name: [np.zeros(3), 0.] for name in forceSensors}
    contact_list = p.getContactPoints()
    for contact in contact_list:
        force_n = contact[9]
        if force_n > 0.:
            force_1 = contact[10]
            force_2 = contact[12]
            surface_n = -1 * np.array(contact[7])
            surface_1 = -1 * np.array(contact[11])
            surface_2 = -1 * np.array(contact[13])
            force = force_n * surface_n + force_1 * surface_1 + force_2 * surface_2
            friction_mu = p.getDynamicsInfo(planeId, -1)[1]
            if contact[4] == -1:
                try: # Get info might produce error when errors in pybullet
                    name = p.getJointInfo(robotId, contact[3])[12].decode("utf-8")
                    if name in forceSensors:
                        f[name] = [0, pin.Force(-force, np.zeros(3)), 2 if np.linalg.norm(force) > 0. else 1 ]
                        s[name] = [-surface_n, friction_mu]
                except:
                    pass
            else:
                try:
                    name = p.getJointInfo(robotId, contact[4])[12].decode("utf-8")
                    if name in forceSensors:
                        f[name] = [0, pin.Force(force, np.zeros(3)), 2 if np.linalg.norm(force) > 0. else 1  ]
                        s[name] = [surface_n, friction_mu]
                except:
                    pass

    return f,s
