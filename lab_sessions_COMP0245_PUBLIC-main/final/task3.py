import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "random_forest" to use Random Forest models
# Random seed for reproducibility
np.random.seed(0)
# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def solve_inverse_kinematics(goal_position,dyn_model):
    # solve approximate inverse kinematics with error less than 1e-3
    # goal_position: 3x1 numpy array
    # return: 7x1 numpy array
    # Initialize the joint angles with zeros
    q = np.zeros(7)
    error = 1
    # Try with current joint angles and update the joint angles
    while error > 1e-3:
        # Compute the current position
        current_position, _ = dyn_model.ComputeFK(q, "panda_link8")
        # Compute the error
        error = np.linalg.norm(goal_position - current_position)
        # Compute the Jacobian
        J = dyn_model.ComputeJacobian(q, "panda_link8",'global')
        # Compute the change in joint angles
        J_pinv = np.linalg.pinv(J.J)
        delta_q = J_pinv[:,:3] @ (goal_position - current_position)
        # Update the joint angles
        q += delta_q
    return q
    
    



     
def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Load all the models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            # Instantiate the model
            model = MLP()
            # Load the saved model
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            # Load the saved Random Forest model
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 10
    goal_positions = []
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir,use_gui=True)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    
    for num,goal_position in enumerate(goal_positions):
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")
        # final_joint_positions = solve_inverse_kinematics(goal_position, dyn_model)
        # Initialize the simulation
        sim.ResetPose()
        current_time = 0  # Initialize current time

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                # Prepare the test input
                test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                # Predict joint positions using the neural network
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
            elif neural_network_or_random_forest == "random_forest":
                # Predict joint positions using the Random Forest
                predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        # Compute qd_des_over_time by numerically differentiating the predicted joint positions
        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        # Clip the joint velocities to the joint limits
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))
        predicted_cartesian_positions_over_time = np.zeros((len(test_time_array), 3))  # Shape: (num_points, 3)
        # mes_cartesian_positions_over_time = np.zeros((len(test_time_array), 3))  # Shape: (num_points, 3)
        # Data collection loop
        q_mes_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)
        qd_mes_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)
        tau_cmd_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)
        while current_time < test_time_array.max():
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

            # Get the index corresponding to the current time
            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            # Get q_des and qd_des_clip from predicted data
            q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
            qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)
            # Compute the desired cartesian position
            predicted_cartesian_positions_over_time[current_index, :], _ = dyn_model.ComputeFK(q_des, controlled_frame_name)
            q_mes_over_time[current_index, :] = q_mes
            qd_mes_over_time[current_index, :] = qd_mes

            # Comptue measured cartesian position
            # mes_cartesian_positions_over_time[current_index, :], _ = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            tau_cmd_over_time[current_index, :] = tau_cmd
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step

        # After the trajectory, compute the final cartesian position
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")
        
        # Save Data for Visualization
        # Save 3D Trajectory
        save_path = './245/final/task3'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        final_joint_positions = solve_inverse_kinematics(goal_position, dyn_model)
        # Save the data
        save_data = {
            'goal_position': goal_position,
            'predicted_cartesian_positions_over_time': predicted_cartesian_positions_over_time,
            'q_des_over_time': predicted_joint_positions_over_time,
            'qd_des_over_time_clipped': qd_des_over_time_clipped,
            'q_mes_over_time': q_mes_over_time,
            'qd_mes_over_time': qd_mes_over_time,
            'final_joint_positions': final_joint_positions,
            'tau_cmd_over_time': tau_cmd_over_time,
            'test_time_array': test_time_array,
            'neural_network_or_random_forest': neural_network_or_random_forest
        }
        save_filename = os.path.join(save_path, f'data_{num}_{neural_network_or_random_forest}.pkl')
        with open(save_filename, 'wb') as f:
            pickle.dump(save_data, f)

        # # Save Path
        # save_path = './245/final/task3'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # # Visualize results
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # Visualize the goal position and the final computed position
        # ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='r', label='Goal Position')
        # ax.scatter(final_cartesian_pos[0], final_cartesian_pos[1], final_cartesian_pos[2], color='b', label='Computed Position')
        # # Visualize the end effector trajectory in cartesian space
        # # Delete the point with value [0, 0, 0] (not the initial point)
        # predicted_cartesian_positions_over_time = delete_error_points(predicted_cartesian_positions_over_time)
        # ax.plot(predicted_cartesian_positions_over_time[:, 0], predicted_cartesian_positions_over_time[:, 1], predicted_cartesian_positions_over_time[:, 2], label='Predicted Trajectory') 
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # fig.savefig(f'{save_path}/goal_position_{goal_position}_{neural_network_or_random_forest}_3d.png')
        # # Visualize x, y, z positions and velocities over time
        # fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        # axs[0].plot(test_time_array, predicted_cartesian_positions_over_time[:, 0], label='Predicted X')
        # axs[0].plot(test_time_array, predicted_cartesian_positions_over_time[:, 1], label='Predicted Y')
        # axs[0].plot(test_time_array, predicted_cartesian_positions_over_time[:, 2], label='Predicted Z')
        # axs[0].set_xlabel('Time (s)')
        # axs[0].set_ylabel('Position (m)')
        # axs[0].legend()
        # axs[0].grid()
        # #
        # axs[1].plot(test_time_array, qd_des_over_time_clipped[:, 0], label='Predicted X Velocity')
        # axs[1].plot(test_time_array, qd_des_over_time_clipped[:, 1], label='Predicted Y Velocity')
        # axs[1].plot(test_time_array, qd_des_over_time_clipped[:, 2], label='Predicted Z Velocity')
        # axs[1].set_xlabel('Time (s)')
        # axs[1].set_ylabel('Velocity (m/s)')
        # axs[1].legend()
        # axs[1].grid()
        # fig.savefig(f'{save_path}/goal_position_{goal_position}_{neural_network_or_random_forest}.png')

        # plt.show()
    


if __name__ == '__main__':
    main()
    print("Done!")