import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        
        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        # if current_time>=0.02:
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)
        regressor_all.append(cur_regressor)
        tau_mes_all.append(tau_mes)

        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    Y = np.linalg.pinv(X:=np.vstack(regressor_all))
    a = Y@(np.hstack(tau_mes_all))
    print(f"A = {a}")
    # TODO compute the metrics for the linear model
    regressor_all.pop(0)
    u_pred = regressor_all@a
    tau_mes_all.pop(0)

    # 计算 R^2
    RSS = np.sum((tau_mes_all - u_pred) ** 2)  # Residual Sum of Squares
    TSS = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)  # Total Sum of Squares
    R_squared = 1 - (RSS / TSS)
    n = len(regressor_all)
    p = regressor_all[0].shape[1]    
    adjusted_r_squared = 1 - ((1 - R_squared) * (n - 1)) / (n - p - 1)


    print("R_Squared:",R_squared)
    print("Adjusted R-squared:", adjusted_r_squared)
    
    # 计算 MSE
    print("Mean squared error: %.2f" % mean_squared_error(tau_mes_all, u_pred))

    # 计算 F 统计量
    F_statistic = ((TSS-RSS) / p) / ((RSS) / (n - p - 1))
    print("F-statistic:", F_statistic)

    # 置信区间
    residual_variance = RSS / (n - p)
    
    cov_matrix = residual_variance * np.linalg.pinv(X.T@X)
    # 计算标准差
    param_standard_errors = np.sqrt(np.abs(np.diag(cov_matrix)))

    # 计算95%置信区间
    t_value = stats.t.ppf(1 - 0.025, n - p)  # for 95% confidence level
    # A置信区间
    confidence_intervals_A = []
    for i, param in enumerate(a):
        lower_bound = param - t_value * param_standard_errors[i]
        upper_bound = param + t_value * param_standard_errors[i]
        confidence_intervals_A.append((lower_bound, upper_bound))

    # print("Confidence intervals for A:")
    # for i, (lower, upper) in enumerate(confidence_intervals_A):
        # print(f"A[{i}] : [{lower}, {upper}]")
    # u置信区间
    u_standard_errors = X @ param_standard_errors  # Predicted torque standard errors based on A

# Calculate confidence intervals for each predicted value u
    confidence_intervals_u_lower = []
    confidence_intervals_u_upper = []
    for i, u_value in enumerate(u_pred):
        lower_bound = u_value - t_value * u_standard_errors[i]
        upper_bound = u_value + t_value * u_standard_errors[i]
        confidence_intervals_u_lower.append(lower_bound) 
        confidence_intervals_u_upper.append((upper_bound)) 
    # print("Confidence intervals for u:")
    # for i, (lower, upper) in enumerate(confidence_intervals_u):
    #     print(f"u[{i}] : [{lower}, {upper}]")

    # TODO plot the  torque prediction error for each joint (optional)
    # fig,ax = plt.subplots(8)
    # err = np.mean(np.abs(tau_mes_all-u_pred),0)
    # ax[0].bar(range(len(err)),err) 
    # ax[0].set_title("Average Torque Error of Each Joint")
    # ax[0].set_xlabel("Joint")
    # ax[0].set_ylabel("Torque Error (N*m)")

    mes = np.array(list(zip(*tau_mes_all)))
    pred = np.array(list(zip(*u_pred)))
    error = pred - mes
    pred_lower = np.array(list(zip(*confidence_intervals_u_lower)))
    pred_upper = np.array(list(zip(*confidence_intervals_u_upper)))
    time_step_range_begin = 0
    time_step_range_end = 5
    
    ## plot torque prediction with noise
    if sim.bot[0].noise_flag:
        noise = sim.bot[0].robot_noise["joint_cov"][0]
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred_lower[i][time_step_range_begin:],'r--',label = "CI lower")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred_upper[i][time_step_range_begin:],'y--',label = "CI upper")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),mes[i][time_step_range_begin:],'g',label = "Torque Mesurement")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred[i][time_step_range_begin:],'b',label = "Torque Prediction")
            plt.title(f"Torque Prediction of Joint{i} with Noise {noise}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction of Joint{i} with Noise {noise}.png")
            plt.clf()
        # plot torque prediction error with noise
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),error[i][time_step_range_begin:],'b',label = "Torque Prediction Error")
            plt.title(f"Torque Prediction Error of Joint{i} with Noise {noise}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction Error of Joint{i} with Noise {noise} from Step {time_step_range_begin}.png")
            plt.clf()
        # plot torque prediction in given range
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,time_step_range_end-time_step_range_begin),pred[i][time_step_range_begin:5],'b',label = "Torque Prediction")
            plt.plot(range(time_step_range_begin,time_step_range_end-time_step_range_begin),mes[i][time_step_range_begin:5],'g',label = "Torque Mesurement")
            plt.title(f"Torque Prediction Error of Joint{i} with Noise {noise}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction of Joint{i} with Noise {noise} of Beginning Timesteps.png")
            plt.clf()
    
# plot torque prediction
    else:
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred_lower[i][time_step_range_begin:],'r--',label = "CI lower")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred_upper[i][time_step_range_begin:],'y--',label = "CI upper")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),mes[i][time_step_range_begin:],'g',label = "Torque Mesurement")
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),pred[i][time_step_range_begin:],'b',label = "Torque Prediction")
            plt.title(f"Torque Prediction of Joint{i}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction of Joint{i}.png")
            plt.clf()
        # plot torque prediction error
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,len(tau_mes_all)),error[i][time_step_range_begin:],'b',label = "Torque Prediction Error")
            plt.title(f"Torque Prediction Error of Joint{i}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction Error of Joint{i} from Step {time_step_range_begin}.png")
            plt.clf()
        # plot torque prediction in given range
        for i in range(7):
            plt.figure(i)
            plt.plot(range(time_step_range_begin,time_step_range_end-time_step_range_begin),pred[i][time_step_range_begin:5],'b',label = "Torque Prediction")
            plt.plot(range(time_step_range_begin,time_step_range_end-time_step_range_begin),mes[i][time_step_range_begin:5],'g',label = "Torque Mesurement")
            plt.title(f"Torque Prediction Error of Joint{i}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Torque {i} (N*m)")
            plt.legend()
            plt.savefig(f"./245/1/Torque Prediction of Joint{i} of Beginning Timesteps.png")
            plt.clf()
        
    # plt.show()   

if __name__ == '__main__':
    main()
