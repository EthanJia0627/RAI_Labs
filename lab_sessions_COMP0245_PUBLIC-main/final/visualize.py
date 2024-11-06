import torch
from utils import create_data_of_Task1, ShallowCorrectorMLP, DeepCorrectorMLP
import matplotlib.pyplot as plt
import pickle
## compare_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = "./245/final/task1/"



def Compare_Models(List_of_Models,title = "",save_fig = False):
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

def Compare_No_Correction(model_params,title = "",save_fig = False):
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

def get_train_losses(model_params):
    if model_params['shallow']:
        model = ShallowCorrectorMLP(model_params['hidden_size']).to(device)
        model_path = f"{save_path}shallow_corrector_mlp_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
    else:
        model = DeepCorrectorMLP(model_params['hidden_layers'],model_params['hidden_size']).to(device)
        model_path = f"{save_path}deep_corrector_mlp_{model_params['hidden_layers']}_{model_params['hidden_size']}_{model_params['activation']}_{model_params['batch_size']}_{model_params['learning_rate']}.pth"
    model.load_state_dict(torch.load(model_path))
    with open(f'{save_path}train_losses_{model_params["hidden_size"]}_{model_params["activation"]}_{model_params["learning_rate"]}.pkl', 'rb') as f:
        train_losses = pickle.load(f)
    return min(train_losses)

train_losses = []
model_list = []
for layer_size in [32,64,96,128]:
    # model_params = {
    #     'shallow': True,
    #     'hidden_size': layer_size,
    #     'activation': 'ReLU',
    #     'batch_size': 32,
    #     'learning_rate': 0.00001,
    #     'name': f'Shallow MLP (hidden size: {layer_size})',
    #     'color': 'r'
    # }
    # model_list.append(model_params)
    model_params = {
        'shallow': False,
        'hidden_layers': 2,
        'hidden_size': [layer_size,layer_size],
        'activation': 'ReLU',
        'batch_size': 32,
        'learning_rate': 0.00001,
        'name': f'Deep MLP (hidden size: {layer_size})',
        'color': 'b'
    }
    model_list.append(model_params)
    train_losses.append(get_train_losses(model_params))
print(train_losses)
# Compare_Models(model_list,title = "Shallow vs Deep MLP with different hidden sizes",save_fig=True)


# model_params = {
#     'shallow': True,
#     'hidden_size': 128,
#     'activation': 'ReLU',
#     'batch_size': 32,
#     'learning_rate': 0.00001,
#     'name': f'Shallow MLP (hidden size: 64)'
# }
# Compare_No_Correction(model_params,title = "Trajectory Tracking with and without Shallow MLP of size 128",save_fig=True)

