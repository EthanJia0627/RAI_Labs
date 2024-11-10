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

def Filter(pre_data_all,new_data, type = "mean", window_size = 10):
    ## The Pre all data is the data that we have before the new data
    ## The new data is the data that we want to deal with
    ## The type is the filter type, we have mean and median
    ## The window_size is the size of the filter window
    ## The pre_all_data is are 2D tensor with shape (n,7) 7 is the number of joints
    ## The new_data is a 2D tensor with shape 7, 7 is the number of joints
    # filter each joint separately
    for i in range(7):
        # get joint data if window size is bigger than the data size
        if window_size > len(pre_data_all):
            return new_data
        # get joint data
        else:
            joint_data = new_data[i]
            # get the previous data
            pre_joint_data = pre_data_all[-window_size:,i]
            # get the filtered data with numpy
            # append the new data to the previous data
            pre_joint_data = np.append(pre_joint_data,joint_data)
            if type == "mean":
                filtered_data = np.mean(pre_joint_data)
            elif type == "median":
                filtered_data = np.median(pre_joint_data)
            # update the new data
            new_data[i] = filtered_data
    return new_data
