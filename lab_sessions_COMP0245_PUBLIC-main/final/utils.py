import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def create_data_of_Task1():
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
    X_tensor = torch.tensor(X, dtype=torch.float32,device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32,device=device).view(-1, 1)

    return X_tensor,Y_tensor,q_target,dot_q_target,t

def Filter(data, type = "mean", window_size = 10):
    ## data is a sequence of values
    ## type is the type of filter to apply
    ## window_size is the size of the window to apply the filter
    # Filter the data for one afer the other
    filtered_data = []
    for i in range(len(data)):
        if type == "mean":
            filtered_data.append(np.mean(data[max(0,i-window_size):i+1]))
        elif type == "median":
            filtered_data.append(np.median(data[max(0,i-window_size):i+1]))
        elif type == "max":
            filtered_data.append(np.max(data[max(0,i-window_size):i+1]))
        elif type == "min":
            filtered_data.append(np.min(data[max(0,i-window_size):i+1]))
    return filtered_data
