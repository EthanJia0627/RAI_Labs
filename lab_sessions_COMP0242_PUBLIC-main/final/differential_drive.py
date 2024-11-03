import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel
from controllability_analisys import is_controllable
from robot_localization_system import *
import numpy as np


save_path = "./242/final"

def get_circle_points(radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_positions = radius * np.sin(angles)
    y_positions = radius * np.cos(angles)
    headings = wrap_angle(-angles) # Tangential directions
    return np.column_stack((x_positions, y_positions-radius, headings))

def get_line_points():
    return np.array([[1, 3,np.pi/2]])

# Parameters for the simulation
update_AB = True
update_QR = True
Terminal = False
Terminal_Methode = "FiniteHorizon"
EKF = True
sensor_type = "RB"
save_fig = False
# Parameters for the MPC controller
Qcoeff_init = np.array([380, 380, 2500])
Rcoeff_init = 3.5
Qcoeff_update = np.array([80, 80, 1200])
Rcoeff_update = 1
Cost_Update_Threshold = 1
Target = get_circle_points(10, 30)
Target = get_line_points()
Target_Update_Threshold = 0.5
N_mpc_init = 20


# global variables

landmarks = np.array([
            [-1, -1],
            [-1, 3],
            [3, -1],
            [3,3],
            [-1,1],
            [1,-1],
            [1,3],
            [3,1],
            [1,1]
        ])

# x_range = np.linspace(-10, 10, num=5)  # 5个x坐标
# y_range = np.linspace(-20, 0, num=5)   # 5个y坐标
# landmarks = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
# landmarks = landmarks[~np.all(landmarks == [0, 0], axis=1)]


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.3, 0.3, 0.1]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.1 ** 2
        self.W_bearing = (np.pi * 0.05 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([0,0,0])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2

map = Map(landmarks)
filter_config = FilterConfiguration()

def landmark_range_observations(x):
    y = []
    C = []
    for lm in map.landmarks:
        # True range measurement (with noise)
        dx = lm[0] - x[0]
        dy = lm[1] - x[1]
        range_meas = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(filter_config.W_range))
        # range_meas = np.sqrt(dx**2 + dy**2) 
       
        y.append(range_meas)

    y = np.array(y)
    return y

def landmark_range_bearing_observations(pos,theta):
        y_range = []
        y_bearing = []
        C = []
        for lm in map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - pos[0]
            dy = lm[1] - pos[1]
            range_meas = np.sqrt(dx**2 + dy**2)
            bearing_meas = np.arctan2(dy,dx) - theta + np.random.normal(0, np.sqrt(filter_config.W_bearing))
            # bearing_meas = np.arctan2(dy,dx) - theta 
            # range_meas = range_true + np.random.normal(0, np.sqrt(W_range))
            # bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))
            y_range.append(range_meas)
            y_bearing.append(bearing_meas)
        y_range = np.array(y_range)
        y_bearing = np.array(y_bearing)
        return y_range,y_bearing


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

   
    # Initialize data storage
    base_pos_all, base_bearing_all = [], []
    base_pos_estimate_all, base_bearing_estimate_all = [],[]
    base_pos_pred_all, base_bearing_pred_all = [],[]
    base_pos_nonoise_all, base_bearing_nonoise_all = [],[]  # for comparison purpose
    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = N_mpc_init

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    # update A,B,C matrices
    # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # you can linearize around the final state and control of the robot (everything zero)
    # or you can linearize around the current state and control of the robot
    # in the second case case you need to update the matrices A and B at each time step
    # and recall everytime the method updateSystemMatrices
    init_pos  = np.array([2.0, 3.0])
    # init_quat = np.array([0,0,0.3827,0.9239])
    theta = 0.0
    init_quat = np.array([0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.array([0.001, 0.001])
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    is_controllable(regulator.A,regulator.B)
    # Define the cost matrices
    Qcoeff = Qcoeff_init
    Rcoeff = Rcoeff_init
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   

    u_mpc = np.zeros(num_controls)

    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    Tre_idx = 0
    if EKF:
        # Create the estimator and start it.
        if sensor_type == "R":
            estimator = RobotEstimator_R(filter_config, map)
        else:
            estimator = RobotEstimator_RB(filter_config, map)
        estimator.start()
        estimator.set_control_input(cur_u_for_linearization)
    while True:


        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction
        if EKF:
            estimator.predict_to(current_time)
            base_pos_pred_all.append(estimator._x_pred[:2])
            base_bearing_pred_all.append(estimator._x_pred[2])
        # Get the measurements from the simulator ###########################################
         # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        if sensor_type == "R":
            y = landmark_range_observations(base_pos_no_noise)
        else:
            y,y_bearing = landmark_range_bearing_observations(base_pos_no_noise,base_bearing_no_noise_)
        # Update the filter with the latest observations
        if EKF:
            if sensor_type == "R":
                estimator.update_from_landmark_range_observations(y)
            else:
                estimator.update_from_landmark_range_bearing_observations(y,y_bearing)
                
        # Get the current state estimate
            x_est,Sigma_est = estimator.estimate()

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
       
   
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        if update_AB:
            cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
            cur_u_for_linearization = u_mpc
            regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
        x0_mpc = np.hstack((base_pos[:2], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        
        if EKF:
            x0_mpc = x_est.copy()
            base_pos_estimate_all.append(x_est[:2])
            base_bearing_estimate_all.append(x_est[2])
        
        if Target is not None:
            x0_mpc -= Target[Tre_idx]
            x0_mpc[2] = wrap_angle(x0_mpc[2])
            # update target if the robot is close to the target
            if (abs(x0_mpc[0:2])<Target_Update_Threshold).all(): 
                Tre_idx += 1
                Qcoeff = Qcoeff_init
                Rcoeff = Rcoeff_init
                if Tre_idx == len(Target):
                    break

        if Terminal:
            regulator.update_P()
            u_mpc = regulator.get_u_DARE(x0_mpc,Terminal_Methode)
        else:
            if update_QR:
                if (abs(x0_mpc)<Cost_Update_Threshold).all(): 
                    Qcoeff = Qcoeff_update
                    Rcoeff = Rcoeff_update
                    # if (abs(x0_mpc)<0.5).any():    
                    #     Qcoeff[np.where(abs(x0_mpc)>0.6)] = 800
                    regulator.setCostMatrices(Qcoeff,Rcoeff)
            u_mpc = regulator.get_u(x0_mpc)
        if EKF:
            estimator.set_control_input(u_mpc)
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)
        base_pos_nonoise_all.append(base_pos_no_noise)
        base_bearing_nonoise_all.append(base_bearing_no_noise_)

        # Update current time
        current_time += time_step
    print("Runtime: ", current_time)

# Plotting 
#add visualization of final x, y, trajectory and theta
    # 提取 x 和 y 坐标
    x_vals = np.array(base_pos_all)[:,0]
    y_vals = np.array(base_pos_all)[:,1]
    plt.figure(figsize=(10, 6))

    plt.plot(x_vals, y_vals, label='Trajectory', color='b', marker='.', linestyle='-', markersize=4)

    if EKF:
        plt.clf()
        x_vals = np.array(base_pos_estimate_all)[:,0]
        y_vals = np.array(base_pos_estimate_all)[:,1]
        plt.plot(x_vals, y_vals, label='Estimated Trajectory', color='g', marker='.', linestyle='-', markersize=4 , zorder=4)
        plt.scatter(x_vals[-1], y_vals[-1], label='Final Position', color='orange', marker='x', s=150, zorder=5)
        # plot nonoise position
        x_vals = np.array(base_pos_nonoise_all)[:,0]
        y_vals = np.array(base_pos_nonoise_all)[:,1]
        plt.plot(x_vals, y_vals, label='No Noise Trajectory', color='b', marker='.', linestyle='-', markersize=4)
        # plot landmarks with triangle
        # Plot landmarks with the label only once
        for idx, lm in enumerate(landmarks):
            if idx == 0:
                plt.scatter(lm[0], lm[1], label='Landmarks', color='r', marker='^', s=50, zorder=5)
            else:
                plt.scatter(lm[0], lm[1], color='r', marker='^', s=50, zorder=5)
            

        ## plot predicted position
        # x_vals = np.array(base_pos_pred_all)[:,0]
        # y_vals = np.array(base_pos_pred_all)[:,1]
        # plt.plot(x_vals, y_vals, label='Predicted Trajectory', color='gray', marker='.', linestyle='-', markersize=4)

    # # 添加方向箭头
    # arrow_scale = 0.3  # 调整箭头的大小
    # for i in [len(base_pos_all)-1]:
    #     x, y , _= base_pos_all[i]
    #     theta = base_bearing_all[i]
    #     dx = arrow_scale * np.cos(theta)  # 根据 θ 计算 x 方向的箭头长度
    #     dy = arrow_scale * np.sin(theta)  # 根据 θ 计算 y 方向的箭头长度
    #     plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='b', ec='b')
    
    ## plot destination with only one label
    for idx, lm in enumerate(Target):
        if idx == 0:
            plt.scatter(lm[0], lm[1], label='Destination', color='r', marker='x', s=150, zorder=5)
        else:
            plt.scatter(lm[0], lm[1], color='r', marker='x', s=150, zorder=5)
    

    
    ## plot final position
    plt.scatter(x_vals[-1], y_vals[-1], label='Final Position', color='orange', marker='x', s=150, zorder=5)
    

    # 设置标签和图例
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.grid()
    ## save figure with parameters in the name
    if save_fig:
        save_path = "./242/final"
        if not update_AB:
            save_path += "/AB_constant"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ## save figure with parameters in the name
            plt.savefig(save_path + f"/trajectory_{Qcoeff_init}_{Rcoeff_init}_{N_mpc_init}.png")
        else:
            save_path += "/AB_update"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if Terminal:
                plt.savefig(save_path + f"/terminal_{Qcoeff_init}_{Rcoeff_init}.png")
            elif update_QR:
                ## save figure with initial and updated parameters in the name
                plt.savefig(save_path + f"/update_QR_{Qcoeff_init}_{Rcoeff_init}_{Qcoeff}_{Rcoeff}_{N_mpc_init}.png")
            else:
                plt.savefig(save_path + f"/{Qcoeff_init}_{Rcoeff_init}_{N_mpc_init}.png")

    # 显示图形
    plt.show()


    

if __name__ == '__main__':
    main()