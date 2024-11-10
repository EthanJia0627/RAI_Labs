import os
import numpy as np
import torch
from utils import create_data_of_Task1, ShallowCorrectorMLP, DeepCorrectorMLP
import matplotlib.pyplot as plt
import pickle
## T1_Compare_Models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = "./245/final/task3/"



def T1_Compare_Models(List_of_Models,title = "",save_fig = False):
    # Constants
    
    m = 1.0  # Mass (kg)
    b = 10  # Friction coefficient
    k_p = 50  # Proportional gain
    k_d = 10   # Derivative gain
    dt = 0.01  # Time step
    num_samples = 1000  # Number of samples in dataset
    plt.figure(figsize=(12, 6))
    for model_params in List_of_Models:
        if model_params['shallow']:
            model = ShallowCorrectorMLP(model_params['hidden_size']).to(device)
            model_path = f"{save_path}shallow_corrector_mlp_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
        else:
            model = DeepCorrectorMLP(model_params['hidden_layers'],model_params['hidden_size']).to(device)
            model_path = f"{save_path}deep_corrector_mlp_{model_params['hidden_layers']}_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
        model.load_state_dict(torch.load(model_path))
        _,_,q_target,dot_q_target,t = create_data_of_Task1()
        q_test = 0
        dot_q_test = 0
        q_real_corrected = []
        for i in range(len(t)):
            # Apply MLP correction
            tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
            inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32,device=device)
            correction = model(inputs.unsqueeze(0)).item()
            ddot_q_corrected =(tau - b * dot_q_test + correction) / m
            dot_q_test += ddot_q_corrected * dt
            q_test += dot_q_test * dt
            q_real_corrected.append(q_test)
        # get MSE
        mse = ((q_target - q_real_corrected) ** 2).mean()
        if model_params["color"] is not None:
            plt.plot(t, q_real_corrected, label=f'{model_params["name"]}'+f' MSE: {mse:.4f}',color=model_params["color"])
        else:
            plt.plot(t, q_real_corrected, label=f'{model_params["name"]}'+f' MSE: {mse:.4f}')
    plt.plot(t, q_target, '--', label='Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title(title)
    plt.legend()
    if save_fig:
        plt.savefig(f"{save_path}{title}.png")
    plt.show()

def T1_Compare_No_Correction(model_params,title = "",save_fig = False):
    # This function is used to compare the performance of the PD controller with and without the MLP correction
    # Constants
    m = 1.0  # Mass (kg)
    b = 10  # Friction coefficient
    k_p = 50  # Proportional gain
    k_d = 10   # Derivative gain
    dt = 0.01  # Time step
    num_samples = 1000  # Number of samples in dataset
    plt.figure(figsize=(12, 6))
    if model_params['shallow']:
        model = ShallowCorrectorMLP(model_params['hidden_size']).to(device)
        model_path = f"{save_path}shallow_corrector_mlp_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
    else:
        model = DeepCorrectorMLP(model_params['hidden_layers'],model_params['hidden_size']).to(device)
        model_path = f"{save_path}deep_corrector_mlp_{model_params['hidden_layers']}_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
    model.load_state_dict(torch.load(model_path))
    _,_,q_target,dot_q_target,t = create_data_of_Task1()
    q_test = 0
    dot_q_test = 0
    q_real = []
    q_real_corrected = []
    # integration with only PD Control
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)

    q_test = 0
    dot_q_test = 0
    for i in range(len(t)):
        # Apply MLP correction
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32,device=device)
        correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected =(tau - b * dot_q_test + correction) / m
        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)
    # get MSE of PD only
    mse_pd = ((q_target - q_real) ** 2).mean()
    # get MSE of PD + MLP
    mse_pd_mlp = ((q_target - q_real_corrected) ** 2).mean()
    plt.plot(t, q_target, 'r-', label='Target')
    plt.plot(t, q_real, 'b--', label='PD Only'+f' MSE: {mse_pd:.4f}')
    plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction'+f' MSE: {mse_pd_mlp:.4f}')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    if save_fig:
        plt.savefig(f"{save_path}{title}.png")
    plt.show()

def T1_get_train_losses(model_params):
    with open(f'{save_path}train_losses_{model_params["hidden_size"]}_{model_params["activation"]}_{model_params["batch_size"]}_{model_params["learning_rate"]}.pkl', 'rb') as f:
        train_losses = pickle.load(f)
    return min(train_losses)

def T1_Compare_Loss(List_of_Models,title = "",save_fig = False):
    for model_params in List_of_Models:
        with open(f'{save_path}train_losses_{model_params["hidden_size"]}_{model_params["activation"]}_{model_params["batch_size"]}_{model_params["learning_rate"]}.pkl', 'rb') as f:
            train_losses = pickle.load(f)
        plt.plot(np.log(train_losses), label=f'{model_params["name"]}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    if save_fig:
        plt.savefig(f"{save_path}{title}.png")

def T1_Loss_heatmap(batch_list,lr_list):
    loss_matrix = np.zeros((len(batch_list),len(lr_list)))
    for i,batch in enumerate(batch_list):
        for j,lr in enumerate(lr_list):
            with open(f'{save_path}train_losses_{[32,32]}_ReLU_{batch}_{lr}.pkl', 'rb') as f:
                train_losses = pickle.load(f)
            loss_matrix[i,j] = min(np.log(train_losses))
    plt.imshow(loss_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Learning Rate')
    plt.xticks(np.arange(len(lr_list)),np.log10(lr_list))
    plt.ylabel('Batch Size')
    plt.yticks(np.arange(len(batch_list)),batch_list)
    plt.title('Loss Heatmap (Deep MLP)')
    plt.show()

def T3_get_data(num,neural_network_or_random_forest):
    save_filename = os.path.join(save_path, f'data_{num}_{neural_network_or_random_forest}.pkl')
    with open(save_filename, 'rb') as f:
        save_data = pickle.load(f)
    num = save_data['num']
    goal_position = save_data['goal_position']
    predicted_cartesian_positions_over_time = save_data['predicted_cartesian_positions_over_time']
    predicted_joint_positions_over_time = save_data['q_des_over_time']
    qd_des_over_time_clipped = save_data['qd_des_over_time_clipped']
    test_time_array = save_data['test_time_array']
    neural_network_or_random_forest = save_data['neural_network_or_random_forest']
    return goal_position,predicted_cartesian_positions_over_time,predicted_joint_positions_over_time,qd_des_over_time_clipped,test_time_array,neural_network_or_random_forest

def T3_delete_error_points(cartesian_positions_over_time):
    delete_indices = []
    for i,point in enumerate(cartesian_positions_over_time):
        if np.all(point == 0):
            delete_indices.append(i)
    cartesian_positions_over_time = np.delete(cartesian_positions_over_time, delete_indices, axis=0)
    return cartesian_positions_over_time


def T3_plot_trajectory():
    for i in range(10):
        fig = plt.figure(figsize=(6, 6))
        # Plot in 3D 
        ax = fig.add_subplot(111, projection='3d')
        for neural_network_or_random_forest in ['neural_network','random_forest']:
            goal,traj,q_des,qd_des,t,model = T3_get_data(i,neural_network_or_random_forest)
            traj = T3_delete_error_points(traj)
            # Plot the predicted Cartesian positions over time in 3D
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-'if model == "neural_network" else "g-", label=f'Predicted Position {model}')
        ax.scatter(goal[0], goal[1], goal[2],'r', label='Goal Position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        ax.legend()
        # Save the plot
        plt.savefig(f"{save_path}trajectory_{i}.png")
#===================================================== Task 1 Visualization=====================================================
# batch_list = [8,16,32,64,128,256,512,1000]
# lr_list = [1.0, 1e-1, 1e-2, 1e-3 , 1e-4,1e-5,1e-6]
# T1_Loss_heatmap(batch_list,lr_list)

# shallow_train_losses = []
# deep_train_losses = []
# model_list = []
# for batch in (batch_list:=[32,64,128,256,1000]):
#     model_params = {
#         'shallow': True,
#         'hidden_size': 32,
#         'activation': 'ReLU',
#         'batch_size': batch,
#         'learning_rate': 0.0001,
#         'name': f'Shallow MLP (batch size: {batch})',
#         'color': 'r'
#     }
#     model_list.append(model_params)
#     shallow_train_losses.append(T1_get_train_losses(model_params))

#     # model_params = {
#     #     'shallow': False,
#     #     'hidden_layers': 2,
#     #     'hidden_size': [32,32],
#     #     'activation': 'ReLU',
#     #     'batch_size': batch,
#     #     'learning_rate': 0.0001,
#     #     'name': f'Deep MLP (batch size: {batch})',
#     #     'color': 'b'
#     # }
#     # model_list.append(model_params)
#     # deep_train_losses.append(T1_get_train_losses(model_params))
# T1_Compare_Loss(model_list,title = "Training Loss with different batch sizes",save_fig=True)


# plt.figure(figsize=(12, 6))
# plt.plot(np.log10(lr_list),np.log(shallow_train_losses), 'r-', label='Shallow MLP with size 128')
# plt.plot(np.log10(lr_list),np.log(deep_train_losses), 'b-', label='Deep MLP with size 128')
# plt.title('Training Loss with different learning rates')
# plt.xlabel('Log10 Learning Rate')
# plt.ylabel('Log Loss')
# plt.legend()
# plt.savefig(f"{save_path}training_loss.png")
# T1_Compare_Models(model_list,title = "Shallow vs Deep MLP with different hidden sizes",save_fig=True)


# model_params = {
#     'shallow': True,
#     'hidden_size': 128,
#     'activation': 'ReLU',
#     'batch_size': 32,
#     'learning_rate': 0.00001,
#     'name': f'Shallow MLP (hidden size: 64)'
# }
# T1_Compare_No_Correction(model_params,title = "Trajectory Tracking with and without Shallow MLP of size 128",save_fig=True)


#===================================================== Task 3 Visualization=====================================================

T3_plot_trajectory()

