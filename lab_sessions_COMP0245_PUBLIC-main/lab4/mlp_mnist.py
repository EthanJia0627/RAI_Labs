import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *


device = "cuda" if torch.cuda.is_available() else "cpu"
# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the dataset
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./245/4', train=True,
                            transform=transform, download=True)
test_dataset = datasets.MNIST(root='./245/4', train=False,
                            transform=transform, download=True)



for batch_size in batch_sizes:
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,pin_memory_device=device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,pin_memory_device=device)
    
    for structure in structures:
        for activation in activations:
            # 2. Model Construction
            class MLP(nn.Module):
                def __init__(self):
                    
                    super(MLP, self).__init__()
                    self.flatten = nn.Flatten()
                    if activation == 'Relu':
                        self.activation = nn.ReLU()
                    elif activation == 'Sigmoid':
                        self.activation = nn.Sigmoid()
                    if structure == '2layer':            
                        self.hidden = nn.Linear(28*28, 128,device=device)  # Input layer to hidden layer
                        self.hidden2 = nn.Linear(128, 64,device=device)  # Input layer to hidden layer
                        self.output = nn.Linear(64, 10,device=device)     # Hidden layer to output layer
                    else:
                        self.hidden = nn.Linear(28*28,structure,device=device)
                        self.output = nn.Linear(structure,10,device=device)
                        self.hidden2 = None
                    self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

                def forward(self, x):
                    x = self.flatten(x)
                    x = self.hidden(x)
                    x = self.activation(x)
                    if self.hidden2:
                        x = self.hidden2(x)
                        x = self.relu(x)
                    x = self.output(x)
                    x = self.softmax(x)
                    return x
            

            for learning_rate in learning_rates:
            # 3. Model Compilation
                for opt in optimizers:
                    for loss_func in loss_funcs:
                        model = MLP()
                        if  loss_func== 'NLLLoss':
                            criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
                        elif loss_func == 'MSELoss':
                            criterion = nn.MSELoss() 
                        elif loss_func == 'MSELoss':
                            criterion = nn.CrossEntropyLoss() 
                        
                        if opt == 'SGD':
                            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                        elif opt == 'Adam':
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        elif opt == 'RMSprop':
                            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)                    
                        # 4. Model Training
                        epochs = 100
                        train_losses = []
                        valid_losses = []
                        train_accuracies = []
                        print(f"Training Model:\n{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}")
                        if load_data(f"/data/{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}"):
                            print("Abort.")
                            continue
                        for epoch in range(epochs):
                            model.train()
                            epoch_loss = 0
                            correct = 0

                            for data, target in train_loader:
                                data = data.to(device)
                                target = target.to(device)
                                optimizer.zero_grad()
                                output = model(data)
                                if loss_func == "NLLLoss":
                                    loss = criterion(output, target)  # target is not one-hot encoded in PyTorch
                                else:
                                    loss = criterion(torch.exp(output), torch.nn.functional.one_hot(target,10).float())
                                loss.backward()
                                optimizer.step()

                                epoch_loss += loss.item()
                                pred = output.argmax(dim=1, keepdim=True)
                                correct += pred.eq(target.view_as(pred)).sum().item()

                            train_losses.append(epoch_loss / len(train_loader))
                            train_accuracy = 100. * correct / len(train_loader.dataset)
                            train_accuracies.append(train_accuracy)
                            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

                        # 5. Model Evaluation
                        model.eval()
                        test_loss = 0
                        correct = 0

                        with torch.no_grad():
                            for data, target in test_loader:
                                data = data.to(device)
                                target = target.to(device)
                                output = model(data)
                                if loss_func == "NLLLoss":
                                    loss = criterion(output, target)  # target is not one-hot encoded in PyTorch
                                else:
                                    loss = criterion(torch.exp(output), torch.nn.functional.one_hot(target,10).float())

                                test_loss += loss.item()
                                pred = output.argmax(dim=1, keepdim=True)
                                correct += pred.eq(target.view_as(pred)).sum().item()

                        test_loss /= len(test_loader)
                        test_accuracy = 100. * correct / len(test_loader.dataset)
                        figure = plt.figure(0)
                        plt.title(f"Training Loss by Epoch\n{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}")
                        plt.plot(range(epochs),train_losses,"b",label = "Training Loss")
                        plt.xlabel("Epoch")
                        plt.ylabel("Training Loss")
                        plt.legend()
                        plt.savefig(savepath+f"/fig/Loss_{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}.png")
                        plt.clf()
                        figure = plt.figure(1)
                        plt.title(f"Accuracy by Epoch\n{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}")
                        plt.plot(range(epochs),train_accuracies,"b",label = "Training Accuracy")
                        plt.xlabel("Epoch")
                        plt.ylabel("Training Accuracy")
                        plt.legend()
                        plt.savefig(savepath+f"/fig/Accuracy_{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}.png")
                        plt.clf()
                        save_data([train_losses,train_accuracies],f"/data/{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}")
                        torch.save(model.state_dict(),savepath+f"/model/{structure}_{loss_func}_{activation}_{opt}_{learning_rate}_{batch_size}.pth")
                        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
                        # plt.show()




