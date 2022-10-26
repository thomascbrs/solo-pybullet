# coding: utf8

# Other modules
from configparser import Interpolation
import numpy as np
# Pinocchio modules
import pinocchio as pin  # Pinocchio library
import example_robot_data
import caracal
from .PD import PD
import crocoddyl
################
#  CONTROLLER ##
################

STEP_LENGTH = 0.3
N_GAITS = 5
TYPE_OF_GAIT = "pace"  # walk, trot, pace, and jump
N_MPC_STEPS = 350
MPC_HORIZON = 125


class Results():
    """ Store the quantities controlled during one iteration.
    ! Using array, not column-wise array.
    """

    def __init__(self) -> None:
        # Current State
        self.x = np.zeros((37)) # State x37

        # Reference values from MPC, updated at dt_mpc
        self.xref = np.zeros((37,))  # State x37
        self.uff = np.zeros((12,)) # torques ref x 12
        self.Kref = np.zeros((12,36))  # Ricatti gain
        self.fref = np.zeros((12,))  # Forces ref, pin.Forces

        self.q_PD = np.zeros((12,))   # Reference joint position for PD, column x(12,1)
        self.qv_PD = np.zeros((12,))  # Reference joint velocity for PD, column x(12,1)
        self.u_PD = np.zeros((12,))  # Reference torques for PD, column x(12,1)

        # Command sent
        self.u_cmd = np.zeros((12,))
        self.u_fb = np.zeros((12,)) # Torques feedeback from Ricatti Gain
        self.u_kpkd = np.zeros((12,)) # Torques due to KpKd
        self.f_dyn = np.zeros((12,)) # Forces obtained using pinocchio contactDynamic

        # Interpolate xref
        self.xref_int = np.zeros((37,)) # Interpolate xref using contact dynamics

class Controller():

    def __init__(self, dt):

        self._dt = dt
        self._dt_mpc = 0.02
        self._kmpc = int(self._dt_mpc / self._dt)
        self._results = Results()
        self.forder = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self._robot = example_robot_data.load("solo12", True)
        self._model = self._robot.model
        self._data = self._model.createData()
        self._model.effortLimit *= 1.
        self._model.velocityLimit *= 2.
        self._model.lowerPositionLimit[7:] = np.array(
            [-1.2, -2., -3.12, -1.2, -2., -3.12, -1.2, -2., -3., -1.2, -2., -3.12])
        self._model.upperPositionLimit[7:] = np.array(
            [1.2, 2., +3.12, +1.2, 2., +3.12, 1.2, 2., +3.12, +1.2, 2., +3.12])
        self._model.velocityLimit[6:] = 80 * np.ones(12)

        # Initial state and its contact placement
        self._q0 = self._model.referenceConfigurations['straight_standing'].copy()
        # Inverse leg orientation
        # for i in range(4):
        #     self._q0[8 + 3*i] = 0.8
        #     self._q0[9 + 3*i] = -1.6
        v0 = pin.utils.zero(self._model.nv)
        x0 = np.concatenate([self._q0, v0])
        data = self._model.createData()
        pin.forwardKinematics(self._model, data, self._q0, v0)
        pin.updateFramePlacements(self._model, data)

        # Initialize parameters for Contact Dynamic estimation
        self.initializeContactDynamic()

        # Define the gait
        gait_generator = caracal.QuadrupedalGaitGenerator(lf="FL_FOOT", lh="HL_FOOT", rf="FR_FOOT", rh="HR_FOOT")
        cs = [None] * (N_GAITS + 1)
        # STEP_LENGTH = [0.0,0.05,0.05,0.4,0.1,0.05,0.05,0.1,0.1,0.1,0.5,0.5]
        # HEIGHT = [0.,0.,0.,0.2,0.,0.,0.]
        # offset = 0.2

        jumps = dict()
        jumps["jump4"] = {
            "STEP_LENGTH": [0.0, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
            "HEIGHT": [0., 0., 0., 0.2, 0., 0., 0.],
            "offset": 0.2,
            "height_jump": 0.2,
            "N_control": [25, 35, 10, 10],  # N_ds, N_ss, N_TO, N_TD, True, True
        }
        jumps["jump3"] = {
            "STEP_LENGTH": [0.0, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
            "HEIGHT": [0., 0., 0., 0.2, 0., 0., 0.],
            "offset": 0.2,
            "height_jump": 0.2,
            "N_control": [25, 30, 10, 10],  # N_ds, N_ss, N_TO, N_TD, True, True
        }
        jumps["jump2"] = {
            "STEP_LENGTH": [0.0, 0.05, 0.05, 0.4, 0.4, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
            "HEIGHT": [0., 0., 0., 0., 0., 0., 0.],
            "offset": 0.12,
            "height_jump": 0.1,
            "N_control": [30, 20, 5, 5],  # N_ds, N_ss, N_TO, N_TD, True, True
        }
        jumps["jump1"] = {
            "STEP_LENGTH": [0.0, 0.05, 0.05, 0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
            "HEIGHT": [0., 0., 0., 0., 0., 0., 0.],
            "offset": 0.12,
            "height_jump": 0.1,
            "N_control": [20, 25, 0, 0],  # N_ds, N_ss, N_TO, N_TD, True, True
        }
        # jumps["jump5"] = {
        #     "STEP_LENGTH": [0.0, 0.8, 0., 0., 0.1, 0.0, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
        #     "HEIGHT": [0., 0., 0., 0., 0., 0., 0.],
        #     "offset": 0.,
        #     "height_jump": 0.3,
        #     "N_control": [30, 45, 0, 0],  # N_ds, N_ss, N_TO, N_TD, True, True
        # }
        jumps["jump5"] = {
            "STEP_LENGTH": [0.0, 0., 0., 0., 0.1, 0.0, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
            "HEIGHT": [0., 0., 0., 0., 0., 0., 0.],
            "offset": 0.,
            "height_jump": 1.,
            "N_control": [30, 65, 0, 0],  # N_ds, N_ss, N_TO, N_TD, True, True
        }


        jump = "jump5"
        STEP_LENGTH = jumps.get(jump)["STEP_LENGTH"]
        HEIGHT = jumps.get(jump)["HEIGHT"]
        offset = jumps.get(jump)["offset"]
        height_jump = jumps.get(jump)["height_jump"]
        N_control = jumps.get(jump)["N_control"]

        sum_front = 0
        sum_hind = 0
        sum_height = 0
        for c in range(N_GAITS + 1):
            cs[c] = dict()
            print("c : ", c)
            for name in ["HL_FOOT", "FL_FOOT", "HR_FOOT", "FR_FOOT"]:
                oMf = data.oMf[self._model.getFrameId(name)]
                if name == "HL_FOOT" or name == "HR_FOOT":
                    if c == 2:
                        cs[c][name] = pin.SE3(
                            oMf.rotation, oMf.translation +
                            np.array([sum_hind + STEP_LENGTH[c] + offset, 0., sum_height + HEIGHT[c]]))
                    else:
                        cs[c][name] = pin.SE3(
                            oMf.rotation,
                            oMf.translation + np.array([sum_hind + STEP_LENGTH[c], 0., sum_height + HEIGHT[c]]))
                else:
                    cs[c][name] = pin.SE3(
                        oMf.rotation,
                        oMf.translation + np.array([sum_front + STEP_LENGTH[c], 0., sum_height + HEIGHT[c]]))

            sum_front += STEP_LENGTH[c]
            sum_height += HEIGHT[c]
            if c != 2:
                sum_hind += STEP_LENGTH[c]
            else:
                sum_hind += STEP_LENGTH[c] + offset

        # Jump 1 --> 4
        # Initial trotting phase
        # c = 0
        # contacts, stepHeight = [cs[c], cs[c + 1]], 0.05
        # gait = gait_generator.trot(contacts, 45, 25, 0, 0, stepHeight, True, False)
        # c += 1
        # contacts, stepHeight = [cs[c], cs[c + 1]], 0.05
        # gait += gait_generator.trot(contacts, 45, 30, 0, 0, stepHeight, False, False)
        # c += 1
        # # Agressive jump
        # contacts, stepHeight = [cs[c], cs[c + 1]], height_jump
        # N_ds, N_ss, N_TO, N_TD = N_control
        # gait += gait_generator.jump_agressive(contacts, N_ds, N_ss, N_TO, N_TD, stepHeight, True, False)
        # # Stand phase after jump
        # c += 1
        # contacts, stepHeight = [cs[c], cs[c + 1]], 0.08
        # gait += gait_generator.stand(None, 150, None, 0, 0, None, False, False)

        # Jump 5
        # standing phase
        c = 0
        contacts, stepHeight = [cs[c], cs[c + 1]], 0.05
        gait = gait_generator.stand(None, 150, None, 0, 0, None, False, False)
        # Agressive jump
        contacts, stepHeight = [cs[c], cs[c + 1]], height_jump
        N_ds, N_ss, N_TO, N_TD = N_control
        gait += gait_generator.jump_agressive(contacts, N_ds, N_ss, N_TO, N_TD, stepHeight, True, False)
        # Stand phase after jump
        c += 1
        contacts, stepHeight = [cs[c], cs[c + 1]], 0.08
        gait += gait_generator.stand(None, 150, None, 0, 0, None, False, False)

        # Create the MPC application
        params = caracal.CaracalParams()
        # Works with and inverse config of the feet
        # jumps["jump5"] = {
        #     "STEP_LENGTH": [0.0, 0.8, 0., 0., 0.1, 0.0, 0.05, 0.1, 0.1, 0.1, 0.5, 0.5],
        #     "HEIGHT": [0., 0., 0., 0., 0., 0., 0.],
        #     "offset": 0.,
        #     "height_jump": 0.3,
        #     "N_control": [30, 45, 0, 0],  # N_ds, N_ss, N_TO, N_TD, True, True
        # }
        # params.withControlGrav = True
        # params.withForceReg = True
        # params.withImpulseReg = True
        # params.withZeroForceReg = True
        # params.withZeroImpulseReg = True
        # params.w_com = 10
        # params.w_impulseReg = 100.
        # params.Qx_BaseRot = 50
        # params.Qx_BaseAngVel = 10.
        # params.Qx_JointVel = 0.5
        # params.Qx_JointPos = 0.4
        # params.w_u = 50
        # params.w_forceReg = 50
        # params.w_xbounds = 1000000000
        # params.baumgarteGains = np.array([0., 50.])
        # params.w_friction = 100
        # params.w_x = 50
        # params.w_ximp = 10
        # params.solverType = caracal.SolverType(1)

        # For normal leg config.
        params.withControlGrav = True
        params.withForceReg = True
        params.withImpulseReg = True
        params.withZeroForceReg = True
        params.withZeroImpulseReg = True
        params.w_com = 10
        params.w_impulseReg = 100.
        params.Qx_BaseRot = 50
        params.Qx_BaseAngVel = 10.
        params.Qx_JointVel = 0.5
        params.Qx_JointPos = 0.4
        params.w_u = 150
        params.w_forceReg = 150
        params.w_xbounds = 1000000000
        params.baumgarteGains = np.array([0., 50.])
        params.w_friction = 100
        params.w_x = 100
        params.w_ximp = 10
        params.solverType = caracal.SolverType(1)

        # params.mu = 0.7
        # params.w_com = 1
        # params.w_swing = 1e6
        # params.w_dswing = 1e4
        # params.w_contact = 1e7
        # params.w_dcontact = 1e2
        # params.w_friction = 10
        # params.min_force = 1
        # params.w_xbounds = 100000000
        # params.w_x = 20
        # params.w_ximp = 10
        # params.w_u = 70
        # params.withControlGrav = True
        # params.withForceReg = True
        # params.withImpulseReg = True
        # params.withZeroForceReg = True
        # params.withZeroImpulseReg = True
        # params.w_forceReg = 70.
        # params.w_impulseReg = 5.
        # params.Qx_BasePos = 0.
        # params.Qx_BaseRot = 50.
        # params.Qx_JointPos = 0.01
        # params.Qx_BaseLinVel = 1.
        # params.Qx_BaseAngVel = 1.
        # params.Qx_JointVel = 1.
        # params.Qximp_BasePos = 1.
        # params.Qximp_BaseRot = 1.
        # params.Qximp_JointPos = 10.  # 0.1 for biped
        # params.Qximp_BaseLinVel = 10.
        # params.Qximp_BaseAngVel = 10.
        # params.Qximp_JointVel = 10.
        # params.baumgarteGains = np.array([0., 50.])
        # params.solverType = caracal.SolverType(1)

        self._mpc = caracal.Caracal(self._q0.copy(), self._model, gait, MPC_HORIZON, params, fwddyn=True)
        self._mpc.start(self._q0, maxiter=1)
        self._xref = self._mpc.get_xopt()  
            # Kd = 0.03# State x37
        self._uff = self._mpc.get_uopt()  # torques ref x 12
        self._Kref = self._mpc.get_Kopt()  # Ricatti gain
        self._fref = self._mpc.get_fopt()  # Forces ref, pin.Forces

        # Used for interpolation
        self._xref_old = self._mpc.get_xopt().copy()  # State x37
        self._xref_old[7:19] = self._q0[7:].copy()
        self._uff_old = self._mpc.get_uopt().copy()  # torques ref x 12

        # Reference values for PD control
        self._q_PD = self._q0[7:].copy()   # Reference joint position for PD, column x(12,1)
        self._qv_PD = np.zeros((12,))  # Reference joint velocity for PD, column x(12,1)
        self._u_PD = np.zeros((12,))  # Reference torques for PD, column x(12,1)

        # Torque command sent
        self._u_cmd = np.zeros((12,))
        self._u_fb = np.zeros((12,)) # Ricatti torque feedback
        self._u_kpkd = np.zeros((12,)) # Torques due to KpKd feeback

        # Current state
        self._x = np.zeros((37,))

        # Ricatti gain
        self._state = crocoddyl.StateMultibody(self._model)

            # Kd = 0.03
        # Interpolate at dt xref from MPC using contact dynamics and torques ref _uff
        self._xref_int = np.zeros((37,))

    def initializeContactDynamic(self):
        # Create the model and data for forward simulation
        self._data = self._model.createData()
        dt = self._dt  # IMPORTANT: use the period of the controller loop
        state = crocoddyl.StateMultibody(self._model)
        actuation = crocoddyl.ActuationModelFloatingBase(state)
        costs = crocoddyl.CostModelSum(state, actuation.nu)
        self._contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
        baumgarteGains = np.array([0., 50.])
        self._contact3DNames = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        for name in self._contact3DNames:
            frameId = state.pinocchio.getFrameId(name)
            contact3D = crocoddyl.ContactModel3D(state, frameId, np.zeros(3), actuation.nu, baumgarteGains)
            self._contacts.addContact(name, contact3D)
        diffModel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, self._contacts, costs, 0., True)
        self._amodel = crocoddyl.IntegratedActionModelEuler(diffModel, dt)
        self._adata = self._amodel.createData()
        self._adata_forces = self._amodel.createData()

    def compute(self, q, qdot, dt, i):
        """ Compute torque command using PD controller.
        """
        # unactuated, [x, y, z] position of the base + [x, y, z, w] orientation of the base (stored as a quaternion)
        # qu = q[:7]
        # actuated, [q1, q2, ..., q8] angular position of the 8 motors
        # qa = q[7:]
        # [v_x, v_y, v_z] linear velocity of the base and [w_x, w_y, w_z] angular velocity of the base along x, y, z axes
        # of the world
        # qu_dot = qdot[:6]
        # qa_dot = qdot[6:]  # angular velocity of the 8 motors

        # Update current state
        self._x[:19] = q[:,0]
        self._x[19:] = qdot[:,0]

        t = i * self._dt
        if t > 2. and i % self._kmpc == 0.:  # Switch to trajectory computed by WB-MPC
            # For linear interpolation
            self._xref_old[:] = self._xref[:]
            self._uff_old[:] = self._uff[:]

            # OPEN-LOOP
            x1 = self._mpc.get_xopt(t=1).copy()
            f1 = self._mpc.get_fopt(t=1).copy()
            self._mpc.step(x1[:self._model.nq], x1[self._model.nq:], f1, n_step=2)

            # CLOSED-LOOP
            # No access to forces, perfect force tracking ?
            # --> Not used in MPC.
            # f1 = self._mpc.get_fopt(t=1)
            # x1 = self._mpc.get_xopt(t=1).copy()
            # q_s = q.reshape((19,))[:]
            # # q_s[:7] = x1[:7]
            # qv_s = qdot.reshape((18,))[:]
            # # qv_s[:6] = x1[19:25]
            # self._mpc.step(q_s,qv_s, f1, n_step=2)

            # Get results from MPC
            self._Kref[:,:] = self._mpc.get_Kopt()
            self._xref[:] = self._mpc.get_xopt()
            self._uff[:] = self._mpc.get_uopt()
            self._Kref[:] = self._mpc.get_Kopt()
            self._fref = self._mpc.get_fopt()

            # Integrate of _xref, restart the state
            self._xref_int[:] = self._xref[:]

        # Interpolations
        self._q_PD[:] = self.interpolation((i % self._kmpc) * self._dt, self._xref_old[7:19], self._xref[7:19])
        self._qv_PD[:] = self.interpolation((i % self._kmpc) * self._dt, self._xref_old[25:], self._xref[25:])
        self._u_PD[:] = self.interpolation((i % self._kmpc) * self._dt, self._uff_old, self._uff)

        # Integrate xref during dt_ using uff torques.
        self._xref_int[:] = self.computeContactDynamics(self._xref_int, self._uff, self._fref)

        # Ricatti gain
        # self._u_fb = np.dot(self._Kref, self._state.diff(self._x, self._xref_int))
        self._u_fb = np.zeros((12,))
        self._u_PD[:] = self._uff[:] + self._u_fb[:]

        # Parameters for the PD controller
        Kp = 8.
        Kd = 0.3
        torque_sat = 2.7  # torque saturation in N.m

        if t > 2.5:
            # self._Kref[:,0:6] = 0.
            # self._u_fb = np.dot(self._Kref, self._state.diff(self._x, self._xref_int))
            self._u_PD[:] = self._uff[:]

            # Use integrated ref from MPC
            self._q_PD[:] = self._xref_int[7:19]
            self._qv_PD[:] = self._xref_int[25:]
            # Integration from
            # tau = np.concatenate([np.zeros(6), self._u_PD + self._u_fb])
            # a = pin.aba(self._model, self._data, self._x[:19], self._x[19:], tau)
            # v = self._x[19:] + a * self._dt
            # q_ = pin.integrate(self._model, self._x[:19], v * self._dt)
            # self._q_PD[:] = q_[7:]
            # self._qv_PD[:] = v[6:]
            # pass
            # self._u_PD[:] = np.zeros(12)
            # self._u_PD[:] = self._uff[:]
            Kp = 8.
            Kd = 0.3

        if t > 4.:
            print("4s")
            # Use integrated ref from MPC
            self._q_PD[:] = self._q0[7:].copy()   # Reference joint position for PD, column x(12,1)
            self._qv_PD[:] = np.zeros((12,))  # Reference joint velocity for PD, column x(12,1)
            # self._q_PD[:] = self._xref_int[7:19]
            # self._qv_PD[:] = self._xref_int[25:]
            self._u_PD[:] = np.zeros(12)
            Kp = 8.
            Kd = 0.3


        # Call the PD controller
        self._u_kpkd[:] = Kp * (self._q_PD - q[7:,0]) + Kd * (self._qv_PD - qdot[6:,0]) # Log purposes
        self._u_cmd[:] = PD(self._q_PD, self._qv_PD, q[7:,0], qdot[6:,0], self._dt, Kp, Kd, torque_sat, self._u_PD)

        if t > 2.5:
            self.computeForces(self._xref_int, self._u_cmd)
            # if t > 5. :
            #     from IPython import embed
            #     embed()

        self.update_results()

        # torques must be a numpy array of shape (12, 1) containing the torques applied to the 12 motors
        return self._u_cmd.reshape((12,1))

    def interpolation(self, t, q_previous, q_des):
        """ Linear interpolation at time t % t_mpc between q_previous and q_des
        """
        b = q_previous
        a = (q_des - q_previous) / (self._dt_mpc)
        return a * t + b

    def update_results(self):
        """ Update results of the controlled quantities for analysis.
        """
        # Current State
        self._results.x[:] = self._x[:] # State x37

        # Reference values from MPC, updated at dt_mpc
        self._results.xref[:] = self._xref[:]  # State x37
        self._results.uff[:] = self._uff[:] # torques ref x 12
        self._results.Kref[:] = self._Kref[:]  # Ricatti gain
        # MPC forces
        for id, name in enumerate(self.forder):
            self._results.fref[3 * id:3 * id + 3] = self._fref.get(name)[1].linear

        self._results.q_PD[:] = self._q_PD[:]   # Reference joint position for PD, column x(12,1)
        self._results.qv_PD[:] = self._qv_PD[:]  # Reference joint velocity for PD, column x(12,1)
        self._results.u_PD[:] = self._u_PD[:]  # Reference torques for PD, column x(12,1)

        self._results.u_cmd[:] = self._u_cmd[:] # Torque command sent
        self._results.u_fb[:] = self._u_fb[:] # Torque feedback, Ricatti Gains.
        self._results.u_kpkd[:] = self._u_kpkd[:] # Command due to KpKd feedback

        self._results.xref_int[:] = self._xref_int[:] # Interpolate xref at dt_loop from MPC using _uff

    def computeForces(self, x, u):
        """ Compute contact forces using Pinocchio ContactDynamic model.
        """
        # Already computed inside compute dynammics
        # self._amodel.calc(self._adata_forces, x, u) # Integrate over self.dt to get contact forces in foot frame.
        for i,name in enumerate(self._contact3DNames):
            self._results.f_dyn[3*i:3*i+3] = self._adata.differential.multibody.contacts.contacts.todict().get(name).f.linear

    def computeContactDynamics(self, x, u, f):
        """ This integrate the next state over dt. --> Interpolate the state xref.
        """
        # Enable active contacts based on a force threshold
        for name in self._amodel.differential.contacts.contacts.todict().keys():
            self._amodel.differential.contacts.changeContactStatus(name, False)
        for name, force in f.items():
            if np.linalg.norm(force[1].vector) > 0.:
                self._amodel.differential.contacts.changeContactStatus(name, True)
        self._amodel.calc(self._adata, x, u)
        return self._adata.xnext
