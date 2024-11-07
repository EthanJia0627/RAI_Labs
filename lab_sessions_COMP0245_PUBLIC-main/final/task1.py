import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import create_data_of_Task1, ShallowCorrectorMLP, DeepCorrectorMLP


def task1(**kwargs):

    ## Task 1: Trajectory Tracking with MLP Correction
    # settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shallow = kwargs.get('shallow', True)
    if shallow:
        hidden_size = kwargs.get('hidden_size', 128)
    else:
        hidden_layers = kwargs.get('hidden_layers', 2)
        hidden_size = kwargs.get('hidden_size', [128,128])
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.00001)
    avtivation = "ReLU"
    save_data = kwargs.get('save_data', True)
    save_path = "./245/final/task1/"

    # Constants
    m = 1.0  # Mass (kg)
    b = 10  # Friction coefficient
    k_p = 50  # Proportional gain
    k_d = 10   # Derivative gain
    dt = 0.01  # Time step
    num_samples = 1000  # Number of samples in dataset

    X_tensor,Y_tensor,q_target,dot_q_target,t = create_data_of_Task1()

    # Dataset and DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ShallowCorrectorMLP Model Definition

    # Model, Loss, Optimizer
    if shallow:
        model = ShallowCorrectorMLP(hidden_size).to(device)
    else:
        model = DeepCorrectorMLP(hidden_layers, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    epochs = 1000
    train_losses = []

    ## skip training if model is already trained
    if save_data:
        if shallow:
            model_path = f"{save_path}shallow_corrector_mlp_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.pth"
        else:
            model_path = f"{save_path}deep_corrector_mlp_{hidden_layers}_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.pth"
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Model already trained with {model_path}")
            # return
        except:
            pass
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')

    # Testing Phase: Simulate trajectory tracking
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

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, 'r-', label='Target')
    plt.plot(t, q_real, 'b--', label='PD Only')
    plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
    plt.title('Trajectory Tracking with and without MLP Correction')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()

    # Plot loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # # Save loss curve
    # if shallow:
    #     plt.savefig(f"{save_path}loss_shallow_corrector_mlp_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.png")
    # else:
    #     plt.savefig(f"{save_path}loss_deep_corrector_mlp_{hidden_layers}_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.png")

    # Save Model and Training Loss with parameters in the filename
    if save_data:
    # Create a folder if it does not exist
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save model 
        if shallow:
            torch.save(model.state_dict(), f'{save_path}shallow_corrector_mlp_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.pth')
        else:
            torch.save(model.state_dict(), f'{save_path}deep_corrector_mlp_{hidden_layers}_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.pth')
        # Save training loss with pickle
        import pickle
        with open(f'{save_path}train_losses_{hidden_size}_{avtivation}_{batch_size}_{learning_rate}.pkl', 'wb') as f:
            pickle.dump(train_losses, f)

# plt.show()






if __name__ == "__main__":
    for learning_rate in [1.0, 1e-1, 1e-2, 1e-3 , 1e-4, 1e-5,1e-6]:
        for batch_size in [8,16,32, 64, 128, 256,512,1000]:
            for shallow in [True, False]:
                if shallow:
                    task1(shallow = True, hidden_size=32, batch_size=batch_size, learning_rate=learning_rate, save_data=True)
                else:
                    task1(shallow = False, hidden_layers=2, hidden_size=[32,32], batch_size=batch_size, learning_rate=learning_rate, save_data=True)