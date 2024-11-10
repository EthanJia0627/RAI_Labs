import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as pl
import joblib  # For saving and loading models

# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = True # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing
learning_curve_flag = False  # Set to True to generate the learning curve, False to skip
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
depths_filename = os.path.join(script_dir, 'average_depths.pkl')  # 文件用于保存 average_depths

# Load data

# Check if the file exists
if not os.path.isfile(filename):
    print(f"Error: File {filename} not found in {script_dir}")
else:
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    time_array = np.array(data['time'])            # Shape: (N,)
    q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
    goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

    # Combine time and goal data to form the input features
    X = np.hstack((time_array.reshape(-1, 1), goal_positions))  # Shape: (N, 4)
    # For the purpose of learning curve analysis, select the first joint's position as y
    y = q_mes_all[:, 0]  # Shape: (N,)

average_depths = []
max_depth_set = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_depth_num = max_depth_set[0]

if training_flag:
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
        goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

        # Combine time and goal data to form the input features
        X = np.hstack((time_array.reshape(-1, 1), goal_positions))  # Shape: (N, 4)

        # Split ratio
        split_ratio = 0.90

        # Initialize lists to hold training and test data for all joints
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        train_mse_list = []
        test_mse_list = []
        for joint_idx in range(7):
            # Extract joint data
            y = q_mes_all[:, joint_idx]  # Shape: (N,)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=split_ratio, shuffle=True, random_state=42
            )

            # Store split data
            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

            # Initialize the Random Forest regressor
            rf_model = RandomForestRegressor(
                n_estimators=100,    # Number of trees
                max_depth=max_depth_num,        # Maximum depth of the tree
                random_state=42,
                n_jobs=-1            # Use all available cores
            )

            # Train the model
            rf_model.fit(X_train, y_train)

            #训练完成后，访问每棵树的深度
            tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
            average_depth = sum(tree_depths) / len(tree_depths)
            average_depths.append(average_depth)  # 将每个关节的平均深度添加到列
            print(f"Joint {joint_idx+1} - Average tree depth: {average_depth}")
    
            # Evaluate on training set
            y_train_pred = rf_model.predict(X_train)
            train_mse = np.mean((y_train - y_train_pred) ** 2)

            # Evaluate on test set
            y_test_pred = rf_model.predict(X_test)
            test_mse = np.mean((y_test - y_test_pred) ** 2)
            # store the MSE values for plotting
            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)

            print(f'\nJoint {joint_idx+1}')
            print(f'Train MSE: {train_mse:.6f}')
            print(f'Test MSE: {test_mse:.6f}')


            # Save the trained model
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            joblib.dump(rf_model, model_filename)
            print(f'Model for Joint {joint_idx+1} saved as {model_filename}')
        
        
            
            # Visualization (if enabled)
            if visualize:
                print(f'Visualizing results for Joint {joint_idx+1}...')

                # Plot true vs predicted positions on the test set
                sorted_indices = np.argsort(X_test[:, 0])
                X_test_sorted = X_test[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                y_test_pred_sorted = y_test_pred[sorted_indices]

                plt.figure(figsize=(10, 5))
                plt.plot(X_test_sorted[:, 0], y_test_sorted, label='True Joint Positions')
                plt.plot(X_test_sorted[:, 0], y_test_pred_sorted, label='Predicted Joint Positions', linestyle='--')
                plt.xlabel('Time (s)')
                plt.ylabel('Joint Position (rad)')
                plt.title(f'Joint {joint_idx+1} Position Prediction on Test Set')
                plt.legend()
                plt.grid(True)
                plt.show()
        # Plot the training and test MSE for each joint in log10 scale
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 8), np.log10(train_mse_list), label='Train MSE (log10 scale)')
        plt.plot(range(1, 8), np.log10(test_mse_list), label='Test MSE (log10 scale)')
        plt.xlabel('Joint Index')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Test MSE for Each Joint')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Training and visualization completed.")
        
        with open(depths_filename, 'wb') as f:
            pickle.dump(average_depths, f)
            print("Average depths saved to file.")
else:
    # 尝试从文件加载 average_depths
    if os.path.isfile(depths_filename):
        with open(depths_filename, 'rb') as f:
            average_depths = pickle.load(f)
            print("Average depths loaded from file.")
    else:
        print("No average depths file found. Set training_flag=True to train and save depths.")        

if learning_curve_flag:
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth_num,
        random_state=42,
        n_jobs=-1
    )

    # 生成学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        rf_model, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, np.log10(train_errors), label='Train MSE (log10 scale)')
    plt.plot(train_sizes, np.log10(val_errors), label='Validation MSE (log10 scale)')
    plt.xlabel('Training set size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve for Random Forest Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 在训练完成后绘制每个关节的平均树深度柱状图
    joints = [f"Joint {i+1}" for i in range(len(average_depths))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, average_depths, color='blue')
    plt.xlabel("Joints")
    plt.ylabel("Average Tree Depth")
    plt.title("Average Tree Depth for Each Joint in Random Forest Model")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 添加数值标签
    for bar, avg_depth in zip(bars, average_depths):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x坐标：柱子的中心
            bar.get_height(),                   # y坐标：柱子的高度（即平均深度值）
            f'{avg_depth:.1f}',                 # 标签内容，保留一位小数
            ha='center', va='bottom'            # 标签居中并显示在柱子上方
        )

    plt.show()

if test_cartesian_accuracy_flag:

    if not training_flag:
        # Load the saved data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
        if not os.path.isfile(filename):
            print(f"Error: File {filename} not found in {script_dir}")
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Extract data
            time_array = np.array(data['time'])            # Shape: (N,)

    # Testing with new goal positions
    print("\nTesting the model with new goal positions...")

    # Load all the models into a list
    models = []
    for joint_idx in range(7):
        # Load the saved model
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The name of the saved model
        model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')

        try:
            rf_model = joblib.load(model_filename)

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print("task_22_goal_pos needs to be run at least once with training_flag=True")
            quit()

        models.append(rf_model)

    # Generate new goal positions
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

    # Generate test time array
    test_time_array = np.linspace(time_array.min(), time_array.max(), 100)  # For example, 100 time steps

    # Initialize the dynamic model
    from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

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

    for goal_position in goal_positions:
        print("\nTesting new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (100, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            # Predict joint positions
            y_pred = models[joint_idx].predict(test_input)  # Shape: (num_points,)
            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = y_pred

        # Get the final predicted joint positions (at the last time step)
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)

        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(f"Predicted joint positions at final time step: {final_predicted_joint_positions}")

        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")

        # Optional: Visualize the cartesian trajectory over time
        if visualize:
            cartesian_positions_over_time = []
            for i in range(len(test_time_array)):
                joint_positions = predicted_joint_positions_over_time[i, :]
                cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
                cartesian_positions_over_time.append(cartesian_pos.copy())

            cartesian_positions_over_time = np.array(cartesian_positions_over_time)  # Shape: (num_points, 3)

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(test_time_array, cartesian_positions_over_time[:, 0], label='X Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 1], label='Y Position')
            plt.plot(test_time_array, cartesian_positions_over_time[:, 2], label='Z Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Cartesian Position (m)')
            plt.title('Predicted Cartesian Positions Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot the trajectory in 3D space
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(cartesian_positions_over_time[:, 0], cartesian_positions_over_time[:, 1], cartesian_positions_over_time[:, 2], label='Predicted Trajectory')
            ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label='Goal Position')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            ax.set_title('Predicted Cartesian Trajectory')
            plt.legend()
            plt.show()
        
        
