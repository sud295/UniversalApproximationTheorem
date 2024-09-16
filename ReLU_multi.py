import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

file = "multi"
data = pd.read_csv(f'data/{file}.csv')

inputs = data[['x', 'y', 'z']].values
outputs = data[['f1', 'f2', 'f3']].values

inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(inputs_tensor, outputs_tensor, test_size=0.2, random_state=42)

class ReLUNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sz, output_size):
        super(ReLUNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer_sz),
            nn.ReLU(),
            nn.Linear(hidden_layer_sz, output_size)
        )
    
    def forward(self, x):
        return self.stack(x)

def train_and_plot(hidden_layer_sz, epochs=1000, lr=0.01):
    input_size = 3
    output_size = 3
    
    model = ReLUNN(input_size, hidden_layer_sz, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(x_train)
        loss = criterion(pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    final_loss = criterion(y_test_pred, y_test).item()

    y_test_pred_np = y_test_pred.numpy()
    y_test_np = y_test.numpy()
    x_test_np = x_test.numpy()

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    input_labels = ['x', 'y', 'z']
    output_labels = ['f1', 'f2', 'f3']
    
    for i in range(3):  
        for j in range(3): 
            axs[i, j].scatter(x_test_np[:, i], y_test_np[:, j], color='blue', label='Actual Data')
            axs[i, j].scatter(x_test_np[:, i], y_test_pred_np[:, j], color='red', label='Predicted Data')
            axs[i, j].set_xlabel(f'{input_labels[i]}')
            axs[i, j].set_ylabel(f'{output_labels[j]}')
            axs[i, j].legend()
            axs[i, j].set_title(f'{input_labels[i]} vs {output_labels[j]}')
    
    plt.suptitle(f'Hidden Layer Size: {hidden_layer_sz}, Final Loss: {final_loss:.2f}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'ReLU_plots/{file}/plot_hidden_{hidden_layer_sz}_loss_{final_loss:.2f}.png')
    plt.close()

if file == "big_poly":
    hidden_layer_sizes = [2000, 4000, 16000, 48000]
else:
    hidden_layer_sizes = [2, 20, 200, 2000]

shutil.rmtree(f"ReLU_plots/{file}")
os.makedirs(f"ReLU_plots/{file}")

for sz in hidden_layer_sizes:
    train_and_plot(hidden_layer_sz=sz)
