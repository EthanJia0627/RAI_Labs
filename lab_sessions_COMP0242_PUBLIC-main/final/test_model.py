import numpy as np
from simulation_and_control import pb, PinWrapper, MotorCommands
import os
import matplotlib.pyplot as plt

def test_robot_response(u_input):
    # 初始化仿真
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    

    cmd = MotorCommands()
    
    # 初始化控制命令
    wheel_base_width = 0.46
    wheel_radius = 0.11
    interface_all_wheels = ["velocity"] * 4

    # 仿真参数
    time_step = sim.GetTimeStep()
    total_time = 500
    num_steps = int(total_time / time_step)
    
    # 数据存储
    positions = []
    times = []
    
    for step in range(num_steps):
        current_time = step * time_step
        
        # 应用输入 u
        v = u_input[0]
        w = u_input[1]
        left_wheel_velocity = (2 * v - w * wheel_base_width) / (2 * wheel_radius)
        right_wheel_velocity = (2 * v + w * wheel_base_width) / (2 * wheel_radius)
        angular_wheels_velocity_cmd = np.array([
            right_wheel_velocity,
            left_wheel_velocity,
            left_wheel_velocity,
            right_wheel_velocity
        ])
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)
        
        # 运行仿真
        sim.Step(cmd, "velocity")
        
        # 记录数据
        pos = sim.GetBasePosition()
        positions.append(pos)
        times.append(current_time)
    
    # 绘制位置图
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.xlabel('X 位置')
    plt.ylabel('Y 位置')
    plt.title('机器人轨迹')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 示例输入 u
    u_input = np.array([0.1, 1])  # [线速度, 角速度]
    test_robot_response(u_input)