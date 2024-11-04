import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from decimal import Decimal
from sklearn.metrics import mean_squared_error, r2_score



# Configuration for the simulation

conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
source_names = ["pybullet"]  # Define the source for dynamic modeling

sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir,use_gui=True)  # Initialize simulation interface
# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names,False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()
init_joint_angles = sim.GetInitMotorAngles()
print(f"Initial joint angles: {init_joint_angles}")
# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    arr = np.load('./242/1/record.npy')
    Kp_vec = arr[2]
    Kd_vec = arr[3]
    
    if kp:
        Kp_vec[joint_id] = kp
        Kd_vec[joint_id] = 0


    kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, q_joint_all ,q_joint_des = [], [], [], [] ,[],[]
    

    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, Kp_vec, Kd_vec)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        q_joint_all.append(q_mes[joint_id])
        q_joint_des.append(q_des[joint_id])
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    if plot:
        plt.figure(0)
        plt.plot(q_joint_all)
        plt.plot(q_joint_des)
        plt.title(f"Joint:{joint_id}\nKp:{Kp_vec[joint_id]}\nKd:{Kd_vec[joint_id]}")
        plt.savefig(f"./242/1/Joint:{joint_id}Kp:{Kp_vec[joint_id]}Kd:{Kd_vec[joint_id]}.png")
        plt.show()
    
    return len(find_peaks(q_joint_all)),q_joint_all,time_step
     
def plot(data,kp):
    plt.plot(data)
    plt.title(f"Joint:{joint_id}\nKp:{kp}")
    plt.savefig(f"./242/1/Joint:{joint_id} Kp:{kp}.png")
    plt.show()
    

def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return xf, power


# TODO Implement the table in thi function



def find_peaks(data):
    # 计算相邻元素之间的差（近似导数）
    derivative = np.diff(data)
    # 查找导数为0或导数符号改变的位置
    idx = np.where(np.diff(np.sign(derivative)) != 0)[0] + 1
    peaks = np.array(data)[idx]

    return peaks

def is_sustained(data):
    peaks = find_peaks(data)[-12:]
    data = (abs(np.diff(peaks)))
    data = data/data[0]
    # if 0.9*(start:=abs(peaks[2]-peaks[1]))<=(end:=abs(peaks[-1]-peaks[-2])) and 1.1*start >= end :
    if np.any(data>1.05) or np.any(data<0.98):
        print(data)
        return False
    # if (error:=mean_squared_error(data,np.ones_like(data)*np.mean(data)))>=5e-12:
    #     return False
    else:
        print(data)
        return True

if __name__ == '__main__':
    arr = np.load('./242/1/record.npy')
    print(arr[2],arr[3])
    joint_id = 6 # Joint ID to tune
    regulation_displacement = 0.5  # Displacement from the initial joint position
    init_gain = Decimal('16.6')
    gain_step = Decimal('0.01')
    max_gain = Decimal('20')
    test_duration=20 # in seconds
    current_gain = init_gain
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    peak_num,data,t = simulate_with_given_pid_values(sim,None,joint_id,regulation_displacement,episode_duration=test_duration,plot=True)

    
    

   