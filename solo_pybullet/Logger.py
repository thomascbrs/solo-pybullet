import numpy as np
import pinocchio as pin
from datetime import datetime
import matplotlib.pyplot as plt


class Logger():

    def __init__(self, dt=0.001, dt_mpc=0.02, load=False, data=None, logSize=60e3):
        self.dt = dt
        self.dt_mpc = dt_mpc
        if load:
            if data is None:
                raise ValueError("Cannot load empty data.")
            self.initialize_load(data)
        else:
            self.logSize = np.int(logSize)
            self.Times = np.linspace(0., 0.001 * (logSize - 1), self.logSize)

            self.q = np.zeros([self.logSize, 12])  # Measured joint position
            self.qv = np.zeros([self.logSize, 12])  # Measured joint velocity

            self.pose = np.zeros([self.logSize, 7]) # Measured pose (position, orientation)
            self.vel = np.zeros([self.logSize, 6])  # Measured pose (position, orientation)

            self.q_PD = np.zeros([self.logSize, 12])  # Joint position tracked in PD --> Interpolation from MPC qvlues
            self.qv_PD = np.zeros([self.logSize, 12])  # Joint velocity tracked in PD --> Interpolation from MPC values

            self.q_mpc = np.zeros([self.logSize, 12])  # Joint position from MPC, constant during dt_mpc
            self.qv_mpc = np.zeros([self.logSize, 12])  # Joint velocity from MPC, constant during dt_mpc

            self.pose_mpc = np.zeros([self.logSize, 7])
            self.vel_mpc = np.zeros([self.logSize, 6])

            self.q_int = np.zeros([self.logSize, 12])  # Joint position integrated from MPC ref at dt_loop, using reference torques uff
            self.qv_int = np.zeros([self.logSize, 12]) # Joint velocity integrated from MPC ref at dt_loop, using reference torques uff

            self.pose_int = np.zeros([self.logSize, 7]) # Base pose integrated from MPC ref at dt_loop, using reference torques uff
            self.vel_int = np.zeros([self.logSize, 6]) # Base velocity integrated from MPC ref at dt_loop, using reference torques uff

            self.forces_mpc = np.zeros([self.logSize, 12]) # Reference forces computed by the MPC
            self.f_pyb = np.zeros([self.logSize, 12]) # Forces measured in Pyb
            self.f_dyn = np.zeros([self.logSize, 12]) # Forces measured with Pinocchio Contact Dynamics
            # Order of the forces
            self.forceSensors = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

            self.u_cmd = np.zeros([self.logSize, 12])
            self.u_PD = np.zeros([self.logSize, 12]) # Torques tracked --> Interpolation from MPC values
            self.u_mpc = np.zeros([self.logSize, 12]) # Torques from MPC, constant during dt_mpc
            self.u_fb = np.zeros([self.logSize, 12]) # Feedback torques based on Ricatti gains
            self.u_kpkd = np.zeros([self.logSize, 12]) # Feedback torques based on KpKd

            # More conveniant, but concatenation of [pose,q,vel,qv]
            self.x = np.zeros([self.logSize, 37]) # Current state
            self.xref = np.zeros([self.logSize, 37]) # Reference state from MPC
            self.xref_int = np.zeros([self.logSize, 37]) # State integrated from reference mpc state and contact dynamic model

            self.Kref = np.zeros([self.logSize,12,36])  # Ricatti gain

        self.initialize_plotNames()
        self.i = 0

    def initialize_plotNames(self):
        self.force_names = [
            "FL_fx", "FL_fy", "FL_fz", "FR_fx", "FR_fy", "FR_fz", "HL_fx", "HL_fy", "HL_fz", "HR_fx", "HR_fy", "HR_fz"
        ]
        self.joints_names = [
            "FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE", "HL_HAA", "HL_HFE", "HL_KFE", "HR_HAA",
            "HR_HFE", "HR_KFE"
        ]
        self.position_names = ["X", "Y", "Z"]
        self.orientation_names = ["Roll", "Pitch", "Yaw"]
        self.vel_lin_names = ["Vx", "Vy", "Vz"]
        self.vel_ang_names = ["Wx", "Wy", "Wz"]

    def initialize_load(self, data):
        """ Initialize logger results from numpy data bag.
        """
        self.q = data.get("q")
        self.qv = data.get("qv")
        self.q_PD = data.get("q_PD")
        self.qv_PD = data.get("qv_PD")
        self.q_mpc = data.get("q_mpc")
        self.qv_mpc = data.get("qv_mpc")
        self.q_int = data.get("q_int")
        self.qv_int = data.get("qv_int")
        self.pose = data.get("pose")
        self.vel = data.get("vel")
        self.pose_mpc = data.get("pose_mpc")
        self.vel_mpc = data.get("vel_mpc")
        self.pose_int = data.get("pose_int")
        self.vel_int = data.get("vel_int")
        self.forces_mpc = data.get("forces_mpc")
        self.f_pyb = data.get("f_pyb")
        self.f_dyn = data.get("f_dyn")
        self.u_cmd = data.get("u_cmd")
        self.u_PD = data.get("u_PD")
        self.u_mpc = data.get("u_mpc")
        self.u_fb = data.get("u_fb")
        self.u_kpkd = data.get("u_kpkd")
        self.Kref = data.get("Kref")
        self.x = data.get("x")
        self.xref = data.get("xref")
        self.xref_int = data.get("xref_int")
        self.logSize = self.q.shape[0]
        self.Times = np.linspace(0., 0.001 * (self.logSize - 1), self.logSize)

    def log(self, results):
        """ Log the results controlled.
        Args :
            - results : Object containing the parameters.
            - i : index simulation, time = i * dt
        """
        # Current state
        self.q[self.i] = results.x[7:19]
        self.qv[self.i] = results.x[25:]
        self.pose[self.i] = results.x[:7]
        self.vel[self.i] = results.x[19:25]

        # PD reference values
        self.q_PD[self.i] = results.q_PD[:]
        self.qv_PD[self.i] = results.qv_PD[:]
        self.u_PD[self.i] = results.u_PD[:]

        # MPC reference values
        self.q_mpc[self.i] = results.xref[7:19]
        self.qv_mpc[self.i] = results.xref[25:]
        self.pose_mpc[self.i] = results.xref[:7]
        self.vel_mpc[self.i] = results.xref[19:25]
        self.forces_mpc[self.i] = results.fref[:]
        self.u_mpc[self.i] = results.uff[:]
        self.u_fb[self.i] = results.u_fb[:]

        # MPC reference values integrated
        self.q_int[self.i] = results.xref_int[7:19]
        self.qv_int[self.i] = results.xref_int[25:]
        self.pose_int[self.i] = results.xref_int[:7]
        self.vel_int[self.i] = results.xref_int[19:25]

        # Torques sent
        self.u_cmd[self.i] = results.u_cmd[:]
        self.u_kpkd[self.i] = results.u_kpkd[:]
        self.f_dyn[self.i] = results.f_dyn[:]

        # Ricatti gain
        self.Kref[self.i] = results.Kref[:,:]

        # States
        self.x[self.i] = results.x[:]
        self.xref[self.i] = results.xref[:]
        self.xref_int[self.i] = results.xref_int[:]

        self.i += 1

    def log_pyb_forces(self,f):
        """ Forces inside pin.Forces object
        """
        for id, name in enumerate(self.forceSensors):
            self.f_pyb[self.i][3 * id:3 * id + 3] = f.get(name)[1].linear


    def save(self, path="/home/thomas_cbrs/Desktop/edin_22/solo-pybullet/log/", fileName="data"):
        date_str = datetime.now().strftime("_%Y_%m_%d_%H_%M")
        # name = path + fileName + date_str + "_" + ".npz"
        name = path + "data1.npz"  # Dev, keep constant for now

        np.savez_compressed(name,
                            q=self.q,
                            qv=self.qv,
                            q_PD=self.q_PD,
                            qv_PD=self.qv_PD,
                            q_mpc=self.q_mpc,
                            qv_mpc=self.qv_mpc,
                            q_int=self.q_int,
                            qv_int=self.qv_int,
                            pose=self.pose,
                            vel=self.vel,
                            pose_mpc=self.pose_mpc,
                            vel_mpc=self.vel_mpc,
                            pose_int=self.pose_int,
                            vel_int=self.vel_int,
                            forces_mpc=self.forces_mpc,
                            f_pyb=self.f_pyb,
                            f_dyn=self.f_dyn,
                            u_cmd=self.u_cmd,
                            u_PD = self.u_PD,
                            u_fb = self.u_fb,
                            u_kpkd= self.u_kpkd,
                            Kref=self.Kref,
                            x=self.x,
                            xref=self.xref,
                            xref_int=self.xref_int,
                            u_mpc=self.u_mpc)
        print("Logger saved.")

    #########################################
    def plot_pose_base(self, minTime, maxTime):
        """ Plot Base position and orientation (Euler angles)
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        # Plot Base Position
        id_ = [1, 3, 5]
        for k in range(3):
            ax = plt.subplot(3, 2, id_[k])
            ax.plot(self.Times[id_min:id_max], self.pose[id_min:id_max, k], label=self.position_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max], self.pose_mpc[id_min:id_max, k], label=self.position_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.pose_int[id_min:id_max, k], label=self.position_names[k] + "_int")
            ax.legend()

        rpy_mes = np.array(
            [pin.rpy.matrixToRpy(pin.Quaternion(pose[3:7]).toRotationMatrix()).tolist() for pose in self.pose])
        rpy_mpc = np.array(
            [pin.rpy.matrixToRpy(pin.Quaternion(pose[3:7]).toRotationMatrix()).tolist() for pose in self.pose_mpc])
        rpy_int = np.array(
            [pin.rpy.matrixToRpy(pin.Quaternion(pose[3:7]).toRotationMatrix()).tolist() for pose in self.pose_int])
        # Plot Base Orientation
        id_ = [2, 4, 6]
        for k in range(3):
            ax = plt.subplot(3, 2, id_[k])
            ax.plot(self.Times[id_min:id_max], rpy_mes[id_min:id_max, k], label=self.orientation_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max], rpy_mpc[id_min:id_max, k], label=self.orientation_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], rpy_int[id_min:id_max, k], label=self.orientation_names[k] + "_int")
            ax.legend()
        plt.suptitle("Base position and Orientation")

    #########################################
    def plot_vel_base(self, minTime, maxTime):
        """ Plot Base velocity
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        # Plot Linear velocity
        id_ = [1, 3, 5]
        for k in range(3):
            ax = plt.subplot(3, 2, id_[k])
            ax.plot(self.Times[id_min:id_max], self.vel[id_min:id_max, k], label=self.vel_lin_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max], self.vel_mpc[id_min:id_max, k], label=self.vel_lin_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.vel_int[id_min:id_max, k], label=self.vel_lin_names[k] + "_int")
            ax.legend()

        # Plot Angular velocity
        id_ = [2, 4, 6]
        for k in range(3):
            ax = plt.subplot(3, 2, id_[k])
            ax.plot(self.Times[id_min:id_max], self.vel[id_min:id_max, 3 + k], label=self.vel_ang_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max],
                    self.vel_mpc[id_min:id_max, 3 + k],
                    label=self.vel_ang_names[k] + "_mpc")
            ax.legend()
        plt.suptitle("Base linear and angular velocity")

    #####################################
    def plot_q(self, minTime, maxTime):
        """ Plot Joint positions.
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(12):
            ax = plt.subplot(4, 3, k + 1)
            ax.plot(self.Times[id_min:id_max], self.q[id_min:id_max, k], label=self.joints_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max], self.q_PD[id_min:id_max, k], label=self.joints_names[k] + "_PDref")
            ax.plot(self.Times[id_min:id_max], self.q_mpc[id_min:id_max, k], label=self.joints_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.q_int[id_min:id_max, k], label=self.joints_names[k] + "_int")
            ax.legend()
        plt.suptitle("Joint angular position [rad]")

    ####################################
    def plot_va(self, minTime, maxTime):
        """ Plot joint velocities.
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(12):
            ax = plt.subplot(4, 3, k + 1)
            ax.plot(self.Times[id_min:id_max], self.qv[id_min:id_max, k], label=self.joints_names[k] + "_mes")
            ax.plot(self.Times[id_min:id_max], self.qv_PD[id_min:id_max, k], label=self.joints_names[k] + "_PDref")
            ax.plot(self.Times[id_min:id_max], self.qv_mpc[id_min:id_max, k], label=self.joints_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.qv_int[id_min:id_max, k], label=self.joints_names[k] + "_int")
            ax.legend()
        plt.suptitle("Joint angular velocity [rad.s-1]")

    #########################################
    def plot_torques(self, minTime, maxTime):
        """ Plot Joint position.
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(12):
            ax = plt.subplot(4, 3, k + 1)
            ax.plot(self.Times[id_min:id_max], self.u_mpc[id_min:id_max, k], label=self.joints_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.u_fb[id_min:id_max, k], label=self.joints_names[k] + "_u_fb")
            ax.plot(self.Times[id_min:id_max], self.u_PD[id_min:id_max, k], label=self.joints_names[k] + "_PDref")
            ax.plot(self.Times[id_min:id_max], self.u_kpkd[id_min:id_max, k], label=self.joints_names[k] + "_KpKd")
            ax.plot(self.Times[id_min:id_max],
                    self.u_cmd[id_min:id_max, k],
                    label=self.joints_names[k] + "_cmd")
            ax.legend()
        plt.suptitle("Torques [N.m-1]")

    def plot_torques_fb(self, minTime, maxTime):
        """ Plot Joint position.
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(12):
            ax = plt.subplot(4, 3, k + 1)
            ax.plot(self.Times[id_min:id_max], self.u_mpc[id_min:id_max, k], label=self.joints_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.u_fb[id_min:id_max, k], label=self.joints_names[k] + "_u_fb")
            ax.plot(self.Times[id_min:id_max], self.u_kpkd[id_min:id_max, k], label=self.joints_names[k] + "_KpKd")
            ax.legend()
        plt.suptitle("Torques FB comparison [N.m-1]")

    #########################################
    def plot_forces(self, minTime, maxTime):
        """ Plot forces.
        """
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(12):
            ax = plt.subplot(4, 3, k + 1)
            ax.plot(self.Times[id_min:id_max], self.forces_mpc[id_min:id_max, k], label=self.force_names[k] + "_mpc")
            ax.plot(self.Times[id_min:id_max], self.f_pyb[id_min:id_max, k], label=self.force_names[k] + "_pyb")
            ax.plot(self.Times[id_min:id_max], self.f_dyn[id_min:id_max, k], label=self.force_names[k] + "_dyn")
            ax.legend()
        plt.suptitle("Forces [N]")

    #########################################
    def plot_Kref(self,minTime, maxTime):
        """ Plot Kref for 1 leg
        """
        # Left front leg
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        for k in range(3):
            ax = plt.subplot(3, 1, k + 1)
            for j in range(36):
                ax.plot(self.Times[id_min:id_max], self.Kref[id_min:id_max, k,j])
        plt.suptitle("Kref Left front leg")

    #########################################
    def plot_Kref_torques(self,minTime, maxTime):
        """ Plot the contribution of Kref in the FB torques. for Left front leg
        """
        import example_robot_data
        import crocoddyl
        robot = example_robot_data.load("solo12", True)
        model = robot.model
        state = crocoddyl.StateMultibody(model)
        # Left front leg
        ID_joint = 1 # Front Left leg Hip 2
        plt.figure()
        id_min = int(minTime / self.dt)
        id_max = int(maxTime / self.dt)
        # tau_ = np.zeros([id_max - id_min, 36])
        # for k in range(id_min, id_max):
        #     xdiff = state.diff(self.x[k], self.xref_int[k])
        #     for j in range(36):
        #         tau_[k-id_min,j] = self.Kref[k,ID_joint,j] * xdiff[j]

        # PLot the graph of the different contribution in the Ricatti feedback term
        # for j in range(36):
        #     plt.plot(self.Times[id_min:id_max], tau_[:,j], label="joint_" + str(j))


        X_ = ["Base", "FL", "FR", "HL", "HR", ]
        tau_ = np.zeros([id_max - id_min, 5])
        for k in range(id_min, id_max):
            xdiff = state.diff(self.x[k], self.xref_int[k])

            # Base contrib
            tau_[k-id_min,0] = self.Kref[k,ID_joint,0:6] @ xdiff[0:6] + self.Kref[k,ID_joint,18:24] @ xdiff[18:24]

            # LEgs
            for j in range(4):
                tau_[k-id_min,j+1] = self.Kref[k,ID_joint,6+3*j:6+3*(j+1)] @ xdiff[6+3*j:6+3*(j+1)] + self.Kref[k,ID_joint,24+3*j:24+3*(j+1)] @ xdiff[24+3*j:24+3*(j+1)]

        for k in range(5):
             plt.plot(self.Times[id_min:id_max], tau_[:,k], label=X_[k])

        plt.plot(self.Times[id_min:id_max], self.u_fb[id_min:id_max,ID_joint], label="state feedback")
        plt.legend()
        plt.suptitle("Kref contribution")

    def plot_all(self, minTime, maxTime):
        self.plot_q(minTime, maxTime)
        self.plot_va(minTime, maxTime)
        self.plot_torques(minTime, maxTime)
        self.plot_torques_fb(minTime, maxTime)
        self.plot_pose_base(minTime, maxTime)
        self.plot_vel_base(minTime, maxTime)
        self.plot_forces(minTime, maxTime)
        plt.show(block=True)

    def plot_Ricatti(self, minTime, maxTime):
        self.plot_Kref(minTime, maxTime)
        self.plot_Kref_torques(minTime, maxTime)
        plt.show(block=True)


if __name__ == "__main__":

    path = "/home/thomas_cbrs/Desktop/edin_22/solo-pybullet/log/"
    fileName = "data1.npz"
    name = path + fileName
    # Interval of time on the plots
    minTime = 2.5
    maxTime = 5.2

    data = np.load(name)
    logger = Logger(0.001, 0.02, True, data)
    # logger.plot_all(minTime, maxTime)
    logger.plot_Ricatti(minTime, maxTime)
