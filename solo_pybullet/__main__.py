# coding: utf8

#####################
#  LOADING MODULES ##
#####################

import time

import pybullet as p  # PyBullet simulator

from solo_pybullet.controller import c_walking_IK, c_walking_ID  # Controller functions
from solo_pybullet.controller_caracal import Controller
# Functions to initialize the simulation and retrieve joints positions/velocities
from solo_pybullet.initialization_simulation import configure_simulation, getPosVelJoints, getContactForces
from solo_pybullet.Logger import Logger

####################
#  INITIALIZATION ##
####################

dt = 0.001  # time step of the simulation
dt_mpc = 0.02
LOGGER = True
# If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the
# simulation)
realTimeSimulation = False
enableGUI = True  # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices = configure_simulation(dt, enableGUI)

controller = Controller(dt)
logger = Logger(dt,dt_mpc)

###############
#  MAIN LOOP ##
###############

for i in range(6000):  # run the simulation during dt * i_max seconds (simulation time)

    # Time at the start of the loop
    if realTimeSimulation:
        t0 = time.clock()

    # Get position and velocity of all joints in PyBullet (free flying base + motors)
    q, qdot = getPosVelJoints(robotId, revoluteJointIndices)

    # Call controller to get torques for all joints
    jointTorques = controller.compute(q, qdot, dt, i)

    if LOGGER:
        logger.log(controller._results)
        # if i > 4000:
        # Get contact forces measured in Pybullet
        f,s = getContactForces(robotId, revoluteJointIndices)
        logger.log_pyb_forces(f)

    # This resets the velocity controller for joints that are not actuated
    p.setJointMotorControlArray(robotId, revoluteJointIndices,p.VELOCITY_CONTROL,forces=[0. for name in revoluteJointIndices])
    # Set control torques for all joints in PyBullet
    p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

    # Compute one step of simulation
    p.stepSimulation()

    # Sleep to get a real time simulation
    if realTimeSimulation:
        t_sleep = dt - (time.clock() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)
# Shut down the PyBullet client
p.disconnect()
if LOGGER:
    logger.save()
    logger.plot_all(2.,8.)
