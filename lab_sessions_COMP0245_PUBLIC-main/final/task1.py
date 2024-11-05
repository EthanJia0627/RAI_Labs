import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


## Task 1: Trajectory Tracking with MLP Correction
# settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shallow = True
if shallow:
    hidden_size = 64
else:
    hidden_layers = 3
    hidden_size = [64, 64, 64]
batch_size = 32
learning_rate = 0.00001
avtivation = "ReLU"
save_data = True
save_path = "./245/final/"


# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ShallowCorrectorMLP Model Definition
class ShallowCorrectorMLP(nn.Module):
    def __init__(self,hidden_size=64):
        super(ShallowCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size),
            # we commneted because it is not shallow if we add more layers
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
# DeepCorrectorMLP Model Definition
class DeepCorrectorMLP(nn.Module):
    def __init__(self,hidden_layers = 2,hidden_size=[64,64]):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(hidden_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size[i-1] if i > 0 else 4, hidden_size[i]),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(hidden_size[-1], 1))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# Model, Loss, Optimizer
if shallow:
    model = ShallowCorrectorMLP(hidden_size)
else:
    model = DeepCorrectorMLP(hidden_layers, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
epochs = 1000
train_losses = []


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
    inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
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
    with open(f'{save_path}train_losses_{hidden_size}_{avtivation}_{learning_rate}.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

plt.show()
